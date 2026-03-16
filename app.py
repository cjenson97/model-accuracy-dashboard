import streamlit as st
import pandas as pd
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
from datetime import datetime, timedelta

st.set_page_config(page_title="Model Accuracy Dashboard", layout="wide")

# ── Connection ────────────────────────────────────────────────────────────────
def get_cursor():
    return connect(
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
        s3_staging_dir=st.secrets["aws"]["S3_STAGING_DIR"],
        region_name="eu-west-2",
        work_group="primary",
        cursor_class=PandasCursor,
    ).cursor()

# ── Data loaders — use partition columns to avoid full scans ──────────────────
@st.cache_data(ttl=600)
def load_feedback_loop(year: str, month: str):
    return get_cursor().execute(f"""
        SELECT
            content_id,
            ai_meta_id,
            score_updated_date,
            created_date,
            document_name,
            jurisdiction,
            vertical,
            ai_score,
            ai_score_reason,
            analyst_score,
            analyst_score_reason,
            status,
            status_id
        FROM scans.feedback_loop_prod
        WHERE year = '{year}' AND month = '{month}'
    """).as_pandas()

@st.cache_data(ttl=600)
def load_scored_updates(year: str, month: str):
    return get_cursor().execute(f"""
        SELECT
            content_id,
            document_title,
            jurisdiction,
            vertical,
            authority,
            score,
            score_reason,
            confidence_score,
            confidence_score_in_words,
            processed_time,
            llm_model
        FROM scans.scored_updates
        WHERE year = '{year}' AND month = '{month}'
    """).as_pandas()

# ── Accuracy helpers ──────────────────────────────────────────────────────────
# A score is "accepted" when analyst_score == ai_score (analyst agreed with model)
# Model accuracy for 3s = 3s accepted / total 3s * 100
# Model accuracy for 1s = (total 1s - 1s accepted) / total 1s * 100

BENCHMARK_3S = 80.0  # update with your real benchmarks
BENCHMARK_1S = 70.0

def compute_summary(df):
    results = []
    for industry, grp in df.groupby("jurisdiction"):
        total_3 = len(grp[grp["ai_score"] == 3])
        total_2 = len(grp[grp["ai_score"] == 2])
        total_1 = len(grp[grp["ai_score"] == 1])
        unscored = len(grp[grp["ai_score"].isna()])

        accepted_3 = len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)])
        accepted_1 = len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)])

        acc_3 = round(accepted_3 / total_3 * 100, 1) if total_3 > 0 else None
        acc_1 = round((total_1 - accepted_1) / total_1 * 100, 1) if total_1 > 0 else None

        pending = len(grp[
            (grp["ai_score"].isin([2, 3])) &
            (grp["analyst_score"].isna())
        ])

        results.append({
            "Industry": industry,
            "Score 3": total_3,
            "Score 2": total_2,
            "Score 1": total_1,
            "Unscored": unscored,
            "3s Accepted": accepted_3,
            "1s Accepted": accepted_1,
            "Accuracy 3s %": acc_3,
            "Accuracy 1s %": acc_1,
            "Pending Review (3s+2s)": pending,
        })
    return pd.DataFrame(results)

def colour_cell(val, benchmark):
    if pd.isna(val) or val is None:
        return ""
    return f"color: {'green' if val >= benchmark else 'red'}; font-weight: bold"

# ── Date controls ─────────────────────────────────────────────────────────────
st.title("📊 Model Accuracy Dashboard")

col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Select month", value=datetime.today().replace(day=1))

year_str  = selected_date.strftime("%Y")
month_str = selected_date.strftime("%m")

try:
    with st.spinner("Loading data..."):
        df_feedback = load_feedback_loop(year_str, month_str)
        df_scored   = load_scored_updates(year_str, month_str)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.caption(f"Showing data for **{selected_date.strftime('%B %Y')}** — "
           f"{len(df_feedback):,} feedback rows | {len(df_scored):,} scored rows")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Weekly Summary", "📈 Accuracy Trends", "🔍 Scored Updates"])

# ── TAB 1: Weekly Summary ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Model Accuracy Summary by Industry")

    if df_feedback.empty:
        st.warning("No data found for the selected month.")
    else:
        summary = compute_summary(df_feedback)

        styled = (
            summary.style
            .applymap(lambda v: colour_cell(v, BENCHMARK_3S), subset=["Accuracy 3s %"])
            .applymap(lambda v: colour_cell(v, BENCHMARK_1S), subset=["Accuracy 1s %"])
            .format({"Accuracy 3s %": "{:.1f}%", "Accuracy 1s %": "{:.1f}%"}, na_rep="-")
        )
        st.dataframe(styled, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Score Distribution by Industry**")
            st.bar_chart(summary.set_index("Industry")[["Score 3","Score 2","Score 1","Unscored"]])
        with col_b:
            st.markdown("**Model Accuracy % vs Benchmark**")
            acc = summary.set_index("Industry")[["Accuracy 3s %","Accuracy 1s %"]].dropna(how="all")
            st.bar_chart(acc)

        st.markdown("**Still Pending Review (3s + 2s)**")
        st.bar_chart(summary.set_index("Industry")[["Pending Review (3s+2s)"]])

# ── TAB 2: Weekly Accuracy Trend ──────────────────────────────────────────────
with tab2:
    st.subheader("Weekly Accuracy Trend")

    if df_feedback.empty:
        st.warning("No data found for the selected month.")
    else:
        df_feedback["week"] = pd.to_datetime(
            df_feedback["score_updated_date"], errors="coerce"
        ).dt.to_period("W").astype(str)

        weekly = []
        for (week, ind), grp in df_feedback.groupby(["week", "jurisdiction"]):
            t3 = len(grp[grp["ai_score"] == 3])
            t1 = len(grp[grp["ai_score"] == 1])
            a3 = len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)])
            a1 = len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)])
            weekly.append({
                "Week": week, "Industry": ind,
                "Accuracy 3s %": round(a3/t3*100, 1) if t3 > 0 else None,
                "Accuracy 1s %": round((t1-a1)/t1*100, 1) if t1 > 0 else None,
            })

        df_w = pd.DataFrame(weekly)
        industries = sorted(df_w["Industry"].dropna().unique())
        sel = st.multiselect("Filter industries", industries, default=industries[:6])
        df_w = df_w[df_w["Industry"].isin(sel)]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Score 3 Accuracy % (benchmark: {BENCHMARK_3S}%)**")
            p3 = df_w.pivot(index="Week", columns="Industry", values="Accuracy 3s %")
            p3["Benchmark"] = BENCHMARK_3S
            st.line_chart(p3)
        with col_b:
            st.markdown(f"**Score 1 Accuracy % (benchmark: {BENCHMARK_1S}%)**")
            p1 = df_w.pivot(index="Week", columns="Industry", values="Accuracy 1s %")
            p1["Benchmark"] = BENCHMARK_1S
            st.line_chart(p1)

# ── TAB 3: Scored Updates Explorer ───────────────────────────────────────────
with tab3:
    st.subheader("Scored Updates Detail")

    if df_scored.empty:
        st.warning("No scored_updates data for the selected month.")
    else:
        industries = ["All"] + sorted(df_scored["jurisdiction"].dropna().unique())
        sel_ind = st.selectbox("Filter by Industry", industries)
        view = df_scored if sel_ind == "All" else df_scored[df_scored["jurisdiction"] == sel_ind]

        st.markdown(f"**Score distribution** ({len(view):,} records)")
        score_counts = view["score"].value_counts().sort_index()
        st.bar_chart(score_counts)

        st.markdown("**Confidence Score Distribution**")
        st.bar_chart(view["confidence_score"].value_counts().sort_index())

        st.dataframe(view[[
            "content_id", "document_title", "jurisdiction", "vertical",
            "score", "confidence_score", "confidence_score_in_words",
            "llm_model", "processed_time"
        ]].sort_values("processed_time", ascending=False), use_container_width=True)
