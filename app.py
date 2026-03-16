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

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_scored_updates(start_date: str, end_date: str):
    return get_cursor().execute(f"""
        SELECT *
        FROM scans.scored_updates
        WHERE score_updated_date BETWEEN '{start_date}' AND '{end_date}'
    """).as_pandas()

@st.cache_data(ttl=600)
def load_feedback_loop(start_date: str, end_date: str):
    return get_cursor().execute(f"""
        SELECT *
        FROM scans.feedback_loop_prod
        WHERE score_updated_date BETWEEN '{start_date}' AND '{end_date}'
    """).as_pandas()

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_accuracy(df):
    """
    Model accuracy for 3s  = 3s accepted / total 3s * 100
    Model accuracy for 1s  = (total 1s - 1s accepted) / total 1s * 100
    """
    results = []
    for industry, grp in df.groupby("jurisdiction"):
        total_3 = len(grp[grp["ai_score"] == 3])
        total_2 = len(grp[grp["ai_score"] == 2])
        total_1 = len(grp[grp["ai_score"] == 1])
        unscored = len(grp[grp["ai_score"].isna()])

        accepted_3 = len(grp[(grp["ai_score"] == 3) & (grp["accepted"] == True)])
        accepted_1 = len(grp[(grp["ai_score"] == 1) & (grp["accepted"] == True)])

        acc_3 = round(accepted_3 / total_3 * 100, 1) if total_3 > 0 else None
        acc_1 = round((total_1 - accepted_1) / total_1 * 100, 1) if total_1 > 0 else None

        pending_review = len(grp[(grp["ai_score"].isin([2, 3])) & (grp["accepted"].isna())])

        results.append({
            "Industry": industry,
            "Total Score 3": total_3,
            "Total Score 2": total_2,
            "Total Score 1": total_1,
            "Unscored": unscored,
            "3s Accepted": accepted_3,
            "1s Accepted": accepted_1,
            "Model Accuracy (3s) %": acc_3,
            "Model Accuracy (1s) %": acc_1,
            "Pending Review (3s+2s)": pending_review,
        })
    return pd.DataFrame(results)

BENCHMARK_3S = 80.0   # replace with real benchmark values
BENCHMARK_1S = 70.0

def style_accuracy(val, benchmark):
    if val is None:
        return ""
    color = "green" if val >= benchmark else "red"
    return f"color: {color}; font-weight: bold"

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📊 Model Accuracy Dashboard")

# Date range selector
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("From", value=datetime.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("To", value=datetime.today())

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

try:
    with st.spinner("Loading data..."):
        df_scored = load_scored_updates(start_str, end_str)
        df_feedback = load_feedback_loop(start_str, end_str)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Weekly Summary", "🔬 Accuracy Deep Dive", "👤 Analyst Breakdown"])

# ── TAB 1: Weekly Summary Stats ───────────────────────────────────────────────
with tab1:
    st.subheader("Weekly Summary by Industry")
    st.caption(f"Period: {start_str} → {end_str}")

    if df_feedback.empty:
        st.warning("No data found for the selected date range.")
    else:
        summary = compute_accuracy(df_feedback)

        # Colour accuracy columns against benchmarks
        def highlight_3s(val):
            return style_accuracy(val, BENCHMARK_3S)

        def highlight_1s(val):
            return style_accuracy(val, BENCHMARK_1S)

        styled = summary.style \
            .applymap(highlight_3s, subset=["Model Accuracy (3s) %"]) \
            .applymap(highlight_1s, subset=["Model Accuracy (1s) %"])

        st.dataframe(styled, use_container_width=True)

        # Bar charts
        st.subheader("Score Distribution by Industry")
        chart_data = summary.set_index("Industry")[["Total Score 3", "Total Score 2", "Total Score 1", "Unscored"]]
        st.bar_chart(chart_data)

        st.subheader("Model Accuracy % by Industry")
        acc_data = summary.set_index("Industry")[["Model Accuracy (3s) %", "Model Accuracy (1s) %"]].dropna(how="all")
        st.bar_chart(acc_data)

        st.subheader("Pending Review (3s + 2s) by Industry")
        pending = summary.set_index("Industry")[["Pending Review (3s+2s)"]]
        st.bar_chart(pending)

# ── TAB 2: Accuracy Deep Dive — weekly trend ──────────────────────────────────
with tab2:
    st.subheader("Model Accuracy — Weekly Trend")

    if df_feedback.empty:
        st.warning("No data found for the selected date range.")
    else:
        # Group by week + industry
        df_feedback["week"] = pd.to_datetime(df_feedback["score_updated_date"]).dt.to_period("W").astype(str)

        weekly = []
        for (week, industry), grp in df_feedback.groupby(["week", "jurisdiction"]):
            total_3 = len(grp[grp["ai_score"] == 3])
            total_1 = len(grp[grp["ai_score"] == 1])
            accepted_3 = len(grp[(grp["ai_score"] == 3) & (grp["accepted"] == True)])
            accepted_1 = len(grp[(grp["ai_score"] == 1) & (grp["accepted"] == True)])
            acc_3 = round(accepted_3 / total_3 * 100, 1) if total_3 > 0 else None
            acc_1 = round((total_1 - accepted_1) / total_1 * 100, 1) if total_1 > 0 else None
            weekly.append({"Week": week, "Industry": industry,
                           "Accuracy 3s %": acc_3, "Accuracy 1s %": acc_1})

        df_weekly = pd.DataFrame(weekly)

        industry_list = sorted(df_weekly["Industry"].dropna().unique())
        selected = st.multiselect("Filter by Industry", industry_list, default=industry_list[:5])
        filtered = df_weekly[df_weekly["Industry"].isin(selected)]

        pivot_3 = filtered.pivot(index="Week", columns="Industry", values="Accuracy 3s %")
        pivot_1 = filtered.pivot(index="Week", columns="Industry", values="Accuracy 1s %")

        st.markdown("**Score 3 Accuracy % vs Benchmark**")
        if not pivot_3.empty:
            benchmark_line = pd.DataFrame({"Benchmark": [BENCHMARK_3S] * len(pivot_3)}, index=pivot_3.index)
            st.line_chart(pd.concat([pivot_3, benchmark_line], axis=1))

        st.markdown("**Score 1 Accuracy % vs Benchmark**")
        if not pivot_1.empty:
            benchmark_line = pd.DataFrame({"Benchmark": [BENCHMARK_1S] * len(pivot_1)}, index=pivot_1.index)
            st.line_chart(pd.concat([pivot_1, benchmark_line], axis=1))

# ── TAB 3: Analyst Breakdown ──────────────────────────────────────────────────
with tab3:
    st.subheader("Per-Analyst Breakdown")
    st.caption("Updates received by score and number published, per analyst")

    # scored_updates is expected to have analyst/user info
    analyst_col = None
    for candidate in ["analyst", "analyst_id", "user_id", "user", "assigned_to"]:
        if candidate in df_scored.columns:
            analyst_col = candidate
            break

    if df_scored.empty:
        st.warning("No scored_updates data for the selected date range.")
    elif analyst_col is None:
        st.warning("Could not find an analyst/user column in scored_updates. Available columns: "
                   + ", ".join(df_scored.columns.tolist()))
        st.dataframe(df_scored.head(20), use_container_width=True)
    else:
        analyst_summary = []
        for analyst, grp in df_scored.groupby(analyst_col):
            row = {"Analyst": analyst}
            for score in [3, 2, 1]:
                row[f"Score {score} Received"] = len(grp[grp["ai_score"] == score])
            row["Unscored"] = len(grp[grp["ai_score"].isna()])
            if "published" in grp.columns:
                row["Published"] = grp["published"].sum()
            analyst_summary.append(row)

        df_analyst = pd.DataFrame(analyst_summary)
        st.dataframe(df_analyst, use_container_width=True)
        st.bar_chart(df_analyst.set_index("Analyst")[["Score 3 Received", "Score 2 Received", "Score 1 Received"]])
