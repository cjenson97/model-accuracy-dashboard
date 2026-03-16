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
def load_feedback_loop(year: str, month: str):
    return get_cursor().execute(f"""
        SELECT
            content_id, ai_meta_id, score_updated_date, created_date,
            document_name, jurisdiction, vertical,
            ai_score, ai_score_reason,
            analyst_score, analyst_score_reason,
            status, status_id
        FROM scans.feedback_loop_prod
        WHERE year = '{year}' AND month = '{month}'
    """).as_pandas()

@st.cache_data(ttl=600)
def load_scored_updates(year: str, month: str):
    return get_cursor().execute(f"""
        SELECT
            content_id, document_title, jurisdiction, vertical, authority,
            score, score_reason, confidence_score, confidence_score_in_words,
            processed_time, llm_model
        FROM scans.scored_updates
        WHERE year = '{year}' AND month = '{month}'
    """).as_pandas()

# ── Accuracy helpers ──────────────────────────────────────────────────────────
BENCHMARK_3S = 80.0
BENCHMARK_1S = 70.0

def compute_summary(df):
    results = []
    for industry, grp in df.groupby("jurisdiction"):
        total_3   = len(grp[grp["ai_score"] == 3])
        total_2   = len(grp[grp["ai_score"] == 2])
        total_1   = len(grp[grp["ai_score"] == 1])
        unscored  = len(grp[grp["ai_score"].isna()])
        accepted_3 = len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)])
        accepted_1 = len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)])
        acc_3 = round(accepted_3 / total_3 * 100, 1) if total_3 > 0 else None
        acc_1 = round((total_1 - accepted_1) / total_1 * 100, 1) if total_1 > 0 else None
        pending = len(grp[(grp["ai_score"].isin([2, 3])) & (grp["analyst_score"].isna())])
        results.append({
            "Industry": industry,
            "Score 3": total_3, "Score 2": total_2,
            "Score 1": total_1, "Unscored": unscored,
            "3s Accepted": accepted_3, "1s Accepted": accepted_1,
            "Accuracy 3s %": acc_3, "Accuracy 1s %": acc_1,
            "Pending Review": pending,
        })
    return pd.DataFrame(results)

def colour_cell(val, benchmark):
    if pd.isna(val) or val is None:
        return ""
    return f"color: {'green' if val >= benchmark else 'red'}; font-weight: bold"

def filter_by_week(df, date_col, week_start, week_end):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return df[(dates >= pd.Timestamp(week_start)) & (dates <= pd.Timestamp(week_end))]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Model Accuracy Dashboard")
st.markdown("Tracks how accurately the AI model scored regulatory updates, "
            "and how often analysts agreed with those scores.")

# ── Date / Period Selector ────────────────────────────────────────────────────
st.subheader("🗓 Select Time Period")
st.caption("Choose how you want to slice the data. The month/year selection loads the data — "
           "the week filter then narrows it down further.")

period_col, year_col, month_col, week_col = st.columns(4)

with year_col:
    year = st.selectbox("Year", options=[str(y) for y in range(2023, datetime.today().year + 1)],
                        index=datetime.today().year - 2023)

with month_col:
    month_names = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    month_name = st.selectbox("Month", options=month_names,
                              index=datetime.today().month - 1)
    month_str = str(month_names.index(month_name) + 1).zfill(2)

# Work out the weeks in the selected month
first_day = datetime(int(year), int(month_str), 1)
last_day = (first_day.replace(month=first_day.month % 12 + 1, day=1)
            if first_day.month < 12
            else first_day.replace(year=first_day.year + 1, month=1, day=1)) - timedelta(days=1)

weeks = []
current = first_day
while current <= last_day:
    week_end = min(current + timedelta(days=6), last_day)
    weeks.append(f"{current.strftime('%d %b')} – {week_end.strftime('%d %b')}")
    current += timedelta(days=7)

with week_col:
    week_option = st.selectbox("Week (optional filter)", options=["All weeks"] + weeks)

with period_col:
    st.metric("Period", f"{month_name} {year}")

# Load data
try:
    with st.spinner("Loading data from Athena..."):
        df_feedback = load_feedback_loop(year, month_str)
        df_scored   = load_scored_updates(year, month_str)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Apply week filter if selected
if week_option != "All weeks":
    week_idx   = weeks.index(week_option)
    week_start = first_day + timedelta(weeks=week_idx)
    week_end   = min(week_start + timedelta(days=6), last_day)
    df_feedback = filter_by_week(df_feedback, "score_updated_date", week_start, week_end)
    df_scored   = filter_by_week(df_scored, "processed_time", week_start, week_end)

st.caption(f"📦 Loaded: **{len(df_feedback):,}** feedback rows | **{len(df_scored):,}** scored rows"
           + (f" — filtered to week: **{week_option}**" if week_option != "All weeks" else ""))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Weekly Summary", "📈 Accuracy Trends", "🔍 Scored Updates"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Weekly Summary
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model Accuracy Summary by Industry")
    st.caption("Source: **feedback_loop_prod** table — this records every update the model scored, "
               "along with what the analyst scored it. This is where accuracy is calculated.")

    if df_feedback.empty:
        st.warning("No data found for the selected period.")
    else:
        summary = compute_summary(df_feedback)

        st.markdown("##### 📊 Full Summary Table")
        st.caption("Each row is one industry. **Accuracy 3s %** = how often the model correctly gave "
                   "a high-relevance (score 3) rating. **Accuracy 1s %** = how often the model correctly "
                   "flagged low-relevance (score 1) updates. Green = above benchmark, Red = below benchmark.")

        styled = (
            summary.style
            .applymap(lambda v: colour_cell(v, BENCHMARK_3S), subset=["Accuracy 3s %"])
            .applymap(lambda v: colour_cell(v, BENCHMARK_1S), subset=["Accuracy 1s %"])
            .format({"Accuracy 3s %": "{:.1f}%", "Accuracy 1s %": "{:.1f}%"}, na_rep="-")
        )
        st.dataframe(styled, use_container_width=True)

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### 📦 Score Distribution by Industry")
            st.caption("How many updates each industry received at each score level. "
                       "Score 3 = high relevance, Score 2 = medium, Score 1 = low relevance, "
                       "Unscored = model did not return a score.")
            st.bar_chart(summary.set_index("Industry")[["Score 3","Score 2","Score 1","Unscored"]])

        with col_b:
            st.markdown("##### 🎯 Model Accuracy % by Industry")
            st.caption(f"Accuracy for 3s and 1s side by side. "
                       f"Benchmarks: 3s = {BENCHMARK_3S}%, 1s = {BENCHMARK_1S}%. "
                       "Industries below benchmark need attention.")
            acc = summary.set_index("Industry")[["Accuracy 3s %","Accuracy 1s %"]].dropna(how="all")
            st.bar_chart(acc)

        st.divider()
        st.markdown("##### ⏳ Updates Still Pending Review (Score 3s and 2s)")
        st.caption("Updates that the model gave a score of 3 or 2 but an analyst has not yet reviewed. "
                   "These are still in the queue — a high number here means analysts are behind on reviewing.")
        st.bar_chart(summary.set_index("Industry")[["Pending Review"]])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Weekly Accuracy Trend
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Weekly Accuracy Trend")
    st.caption("Source: **feedback_loop_prod** — shows how model accuracy changed week by week within "
               "the selected month. Useful for spotting if accuracy is improving or declining over time.")

    if df_feedback.empty:
        st.warning("No data found for the selected period.")
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
        sel = st.multiselect("Filter by industry", industries, default=industries[:6])
        df_w = df_w[df_w["Industry"].isin(sel)]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"##### 📈 Score 3 Accuracy % per Week")
            st.caption(f"Each line is one industry. The dotted benchmark line is {BENCHMARK_3S}%. "
                       "Lines below the benchmark mean the model is over-scoring updates as high-relevance.")
            p3 = df_w.pivot(index="Week", columns="Industry", values="Accuracy 3s %")
            p3["── Benchmark"] = BENCHMARK_3S
            st.line_chart(p3)

        with col_b:
            st.markdown(f"##### 📉 Score 1 Accuracy % per Week")
            st.caption(f"Each line is one industry. The dotted benchmark line is {BENCHMARK_1S}%. "
                       "Lines below benchmark mean the model is incorrectly flagging updates as low-relevance.")
            p1 = df_w.pivot(index="Week", columns="Industry", values="Accuracy 1s %")
            p1["── Benchmark"] = BENCHMARK_1S
            st.line_chart(p1)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Scored Updates Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Scored Updates Detail")
    st.caption("Source: **scored_updates** table — this is the raw log of every update the model processed "
               "and scored. It does not include analyst feedback; it shows what the model saw and decided. "
               "Use this to explore individual updates, check confidence levels, or audit the model's output.")

    if df_scored.empty:
        st.warning("No scored_updates data for the selected period.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### 📊 Score Distribution")
            st.caption("How the model distributed scores across all updates. "
                       "A healthy distribution should have most updates at score 1 or 3 "
                       "with fewer at score 2 (medium relevance is harder to judge).")
            st.bar_chart(df_scored["score"].value_counts().sort_index())

        with col_b:
            st.markdown("##### 🔒 Confidence Score Distribution")
            st.caption("How confident the model was in each score. "
                       "Higher confidence means the model was certain. "
                       "Low confidence scores alongside incorrect ratings can indicate model drift.")
            st.bar_chart(df_scored["confidence_score"].value_counts().sort_index())

        st.divider()
        industries = ["All"] + sorted(df_scored["jurisdiction"].dropna().unique())
        sel_ind = st.selectbox("Filter table by Industry", industries)
        view = df_scored if sel_ind == "All" else df_scored[df_scored["jurisdiction"] == sel_ind]

        st.markdown(f"##### 🗂 Raw Records ({len(view):,} updates)")
        st.caption("Full list of updates scored by the model in this period. "
                   "Sorted most recent first.")
        st.dataframe(
            view[[
                "content_id","document_title","jurisdiction","vertical",
                "score","confidence_score","confidence_score_in_words",
                "llm_model","processed_time"
            ]].sort_values("processed_time", ascending=False),
            use_container_width=True
        )
