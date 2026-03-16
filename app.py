import streamlit as st
import pandas as pd
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
from datetime import datetime, timedelta

st.set_page_config(page_title="Model Accuracy Dashboard", layout="wide")

SPREADSHEET_ID = "1B4dxnGuxZTkAPEdFgp5YQ7gui5Sa5UERkIZiF5Fw54E"
BENCHMARK_3S = 80.0
BENCHMARK_1S = 70.0

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

@st.cache_data(ttl=3600)
def load_master_sources():
    """Load the Documents and URLs tab from the Master Sources Google Sheet."""
    url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid=198188912"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.warning(f"Could not load Master Sources sheet: {e}")
        return pd.DataFrame()

# ── Helpers ───────────────────────────────────────────────────────────────────
def filter_by_week(df, date_col, week_start, week_end):
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    return df[
        (dates >= pd.Timestamp(week_start).tz_localize(None)) &
        (dates <= pd.Timestamp(week_end).tz_localize(None))
    ]

def compute_summary(df):
    results = []
    for industry, grp in df.groupby("jurisdiction"):
        total_3    = len(grp[grp["ai_score"] == 3])
        total_2    = len(grp[grp["ai_score"] == 2])
        total_1    = len(grp[grp["ai_score"] == 1])
        unscored   = len(grp[grp["ai_score"].isna()])
        accepted_3 = len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)])
        accepted_1 = len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)])
        acc_3      = round(accepted_3 / total_3 * 100, 1) if total_3 > 0 else None
        acc_1      = round((total_1 - accepted_1) / total_1 * 100, 1) if total_1 > 0 else None
        pending    = len(grp[(grp["ai_score"].isin([2, 3])) & (grp["analyst_score"].isna())])
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

def compute_source_accuracy(df_feedback, df_sources):
    """
    Cross-reference feedback_loop_prod with the Master Sources sheet.
    Groups feedback by jurisdiction + document_name and computes accuracy per source.
    """
    if df_feedback.empty or df_sources.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Normalise join keys
    df_feedback = df_feedback.copy()
    df_feedback["jurisdiction_key"] = df_feedback["jurisdiction"].str.strip().str.lower()

    # Compute accuracy per (jurisdiction, document_name) combination
    source_stats = []
    group_cols = ["jurisdiction", "document_name"] if "document_name" in df_feedback.columns else ["jurisdiction"]

    for keys, grp in df_feedback.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)

        total = len(grp)
        reviewed = grp["analyst_score"].notna().sum()
        if reviewed == 0:
            continue

        total_3 = len(grp[grp["ai_score"] == 3])
        total_1 = len(grp[grp["ai_score"] == 1])
        accepted_3 = len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)])
        accepted_1 = len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)])

        # Overall agreement: analyst score == ai score
        agreed = len(grp[grp["ai_score"] == grp["analyst_score"]])
        agreement_pct = round(agreed / reviewed * 100, 1)

        acc_3 = round(accepted_3 / total_3 * 100, 1) if total_3 > 0 else None
        acc_1 = round((total_1 - accepted_1) / total_1 * 100, 1) if total_1 > 0 else None

        row = {
            "Jurisdiction": keys[0],
            "Document / Source": keys[1] if len(keys) > 1 else "-",
            "Total Updates": total,
            "Analyst Reviewed": int(reviewed),
            "Overall Agreement %": agreement_pct,
            "Accuracy 3s %": acc_3,
            "Accuracy 1s %": acc_1,
        }
        source_stats.append(row)

    if not source_stats:
        return pd.DataFrame(), pd.DataFrame()

    df_stats = pd.DataFrame(source_stats).sort_values("Overall Agreement %", ascending=False)

    # Try to enrich with master source metadata (Industry, Authority, Banding)
    if not df_sources.empty:
        # Detect column names flexibly
        jx_col = next((c for c in df_sources.columns if "jurisd" in c.lower()), None)
        auth_col = next((c for c in df_sources.columns if "author" in c.lower()), None)
        band_col = next((c for c in df_sources.columns if "band" in c.lower()), None)
        ind_col = next((c for c in df_sources.columns if "industr" in c.lower()), None)

        if jx_col:
            merge_cols = {jx_col: "Jurisdiction"}
            extra_cols = {}
            if auth_col:  extra_cols[auth_col] = "Authority"
            if band_col:  extra_cols[band_col] = "Banding"
            if ind_col:   extra_cols[ind_col]  = "Industry"

            src_slim = df_sources[[jx_col] + list(extra_cols.keys())].drop_duplicates(subset=[jx_col])
            src_slim = src_slim.rename(columns={**merge_cols, **extra_cols})
            src_slim["Jurisdiction"] = src_slim["Jurisdiction"].str.strip().str.lower()
            df_stats["_jx_key"] = df_stats["Jurisdiction"].str.strip().str.lower()
            df_stats = df_stats.merge(src_slim, left_on="_jx_key", right_on="Jurisdiction",
                                      how="left", suffixes=("", "_src"))
            df_stats = df_stats.drop(columns=["_jx_key", "Jurisdiction_src"], errors="ignore")

    # Top 10 reliably scored = highest agreement %, min 5 reviewed
    reliable = (df_stats[df_stats["Analyst Reviewed"] >= 5]
                .sort_values("Overall Agreement %", ascending=False)
                .head(10))

    # Top 10 poorly scored = lowest agreement %, min 5 reviewed
    poor = (df_stats[df_stats["Analyst Reviewed"] >= 5]
            .sort_values("Overall Agreement %", ascending=True)
            .head(10))

    return reliable, poor

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Model Accuracy Dashboard")
st.markdown(
    "Tracks how accurately the AI model scored regulatory updates, "
    "and how often analysts agreed with those scores."
)

# ── Date / Period Selector ────────────────────────────────────────────────────
st.subheader("🗓 Select Time Period")
st.caption(
    "Choose a year and month to load data. Optionally filter down to a specific week. "
    "Data is loaded fresh every 10 minutes."
)

period_col, year_col, month_col, week_col = st.columns(4)

with year_col:
    year = st.selectbox(
        "Year",
        options=[str(y) for y in range(2023, datetime.today().year + 1)],
        index=datetime.today().year - 2023,
    )

with month_col:
    month_names = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December",
    ]
    month_name = st.selectbox("Month", options=month_names, index=datetime.today().month - 1)
    month_str  = str(month_names.index(month_name) + 1).zfill(2)

first_day = datetime(int(year), int(month_str), 1)
if first_day.month == 12:
    last_day = first_day.replace(year=first_day.year + 1, month=1, day=1) - timedelta(days=1)
else:
    last_day = first_day.replace(month=first_day.month + 1, day=1) - timedelta(days=1)

weeks = []
current = first_day
while current <= last_day:
    week_end_day = min(current + timedelta(days=6), last_day)
    weeks.append(f"{current.strftime('%d %b')} – {week_end_day.strftime('%d %b')}")
    current += timedelta(days=7)

with week_col:
    week_option = st.selectbox("Week (optional filter)", options=["All weeks"] + weeks)

with period_col:
    st.metric("Period", f"{month_name[:3]} {year}")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading data from Athena..."):
        df_feedback = load_feedback_loop(year, month_str)
        df_scored   = load_scored_updates(year, month_str)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

with st.spinner("Loading Master Sources sheet..."):
    df_sources = load_master_sources()

if week_option != "All weeks":
    week_idx   = weeks.index(week_option)
    week_start = first_day + timedelta(weeks=week_idx)
    week_end   = min(week_start + timedelta(days=6), last_day)
    df_feedback = filter_by_week(df_feedback, "score_updated_date", week_start, week_end)
    df_scored   = filter_by_week(df_scored,   "processed_time",     week_start, week_end)

period_label = f"week of {week_option}" if week_option != "All weeks" else f"{month_name} {year}"
st.caption(
    f"📦 **{len(df_feedback):,}** feedback rows | **{len(df_scored):,}** scored rows "
    f"| **{len(df_sources):,}** master source rows — period: **{period_label}**"
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Weekly Summary",
    "📈 Accuracy Trends",
    "🔍 Scored Updates",
    "🔀 Source Cross-Reference",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Weekly Summary
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model Accuracy Summary by Industry")
    st.caption(
        "Source: **feedback_loop_prod** — records every update the model scored "
        "along with the analyst's score. Accuracy is calculated by comparing the two."
    )

    if df_feedback.empty:
        st.warning("No data found for the selected period.")
    else:
        summary = compute_summary(df_feedback)

        st.markdown("##### 📊 Full Summary Table")
        st.caption(
            "Each row is one industry. "
            "**Score 3/2/1** = how many updates the model gave each rating. "
            "**Accuracy 3s %** = % of score-3 ratings the analyst agreed with. "
            "**Accuracy 1s %** = % of score-1 ratings the analyst agreed were low relevance. "
            "**Pending Review** = updates not yet reviewed by an analyst. "
            "🟢 Green = at or above benchmark. 🔴 Red = below benchmark."
        )
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
            st.caption(
                "Volume of updates at each score level per industry. "
                "Score 3 = high relevance, Score 2 = medium, Score 1 = low relevance, "
                "Unscored = model returned no score."
            )
            st.bar_chart(summary.set_index("Industry")[["Score 3","Score 2","Score 1","Unscored"]])

        with col_b:
            st.markdown("##### 🎯 Model Accuracy % by Industry")
            st.caption(
                f"3s and 1s accuracy side by side. "
                f"Benchmarks: 3s = {BENCHMARK_3S}%, 1s = {BENCHMARK_1S}%. "
                "Industries below benchmark need closer attention."
            )
            acc = summary.set_index("Industry")[["Accuracy 3s %","Accuracy 1s %"]].dropna(how="all")
            st.bar_chart(acc)

        st.divider()
        st.markdown("##### ⏳ Updates Still Pending Analyst Review")
        st.caption(
            "Score-3 and score-2 updates an analyst has not yet reviewed. "
            "A large bar means a backlog — accuracy figures may be incomplete."
        )
        st.bar_chart(summary.set_index("Industry")[["Pending Review"]])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Weekly Accuracy Trend
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Weekly Accuracy Trend")
    st.caption(
        "Source: **feedback_loop_prod** — shows how model accuracy changed week by week "
        "within the selected month. Useful for spotting if accuracy is improving or declining."
    )

    if df_feedback.empty:
        st.warning("No data found for the selected period.")
    else:
        df_feedback["week"] = (
            pd.to_datetime(df_feedback["score_updated_date"], errors="coerce")
            .dt.tz_localize(None)
            .dt.to_period("W")
            .astype(str)
        )

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
        if df_w.empty:
            st.warning("Not enough data to build weekly trends.")
        else:
            industries = sorted(df_w["Industry"].dropna().unique())
            sel = st.multiselect("Filter by industry", industries, default=industries[:6])
            df_w = df_w[df_w["Industry"].isin(sel)]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"##### 📈 Score 3 Accuracy % per Week")
                st.caption(
                    f"Each line = one industry over time. "
                    f"Benchmark line = {BENCHMARK_3S}%. "
                    "Below benchmark = model over-scoring updates as high relevance."
                )
                p3 = df_w.pivot(index="Week", columns="Industry", values="Accuracy 3s %")
                p3["── Benchmark"] = BENCHMARK_3S
                st.line_chart(p3)

            with col_b:
                st.markdown(f"##### 📉 Score 1 Accuracy % per Week")
                st.caption(
                    f"Each line = one industry over time. "
                    f"Benchmark line = {BENCHMARK_1S}%. "
                    "Below benchmark = model incorrectly flagging updates as low relevance."
                )
                p1 = df_w.pivot(index="Week", columns="Industry", values="Accuracy 1s %")
                p1["── Benchmark"] = BENCHMARK_1S
                st.line_chart(p1)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Scored Updates Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Scored Updates Detail")
    st.caption(
        "Source: **scored_updates** table — the raw log of every update the model processed. "
        "Does not include analyst feedback. Shows what the model scored and how confident it was."
    )

    if df_scored.empty:
        st.warning("No scored_updates data for the selected period.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 📊 Score Distribution")
            st.caption(
                "How the model distributed scores. Score 3 = high relevance, "
                "2 = medium, 1 = low."
            )
            st.bar_chart(df_scored["score"].value_counts().sort_index())

        with col_b:
            st.markdown("##### 🔒 Confidence Score Distribution")
            st.caption(
                "How confident the model was in each score. "
                "Low confidence alongside incorrect scores may indicate model drift."
            )
            st.bar_chart(df_scored["confidence_score"].value_counts().sort_index())

        st.divider()
        industries = ["All"] + sorted(df_scored["jurisdiction"].dropna().unique())
        sel_ind = st.selectbox("Filter table by Industry", industries)
        view = df_scored if sel_ind == "All" else df_scored[df_scored["jurisdiction"] == sel_ind]

        st.markdown(f"##### 🗂 Raw Records ({len(view):,} updates)")
        st.caption("Full list of updates scored by the model, sorted most recent first.")
        st.dataframe(
            view[[
                "content_id","document_title","jurisdiction","vertical",
                "score","confidence_score","confidence_score_in_words",
                "llm_model","processed_time",
            ]].sort_values("processed_time", ascending=False),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Source Cross-Reference
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔀 Source Cross-Reference: Reliable vs Unreliable Scoring")
    st.caption(
        "Cross-references **feedback_loop_prod** with the **Master Sources sheet** "
        "(Documents and URLs tab). Groups updates by jurisdiction and document, "
        "calculates how often the analyst agreed with the model score, "
        "and identifies which sources are reliably or poorly scored. "
        "Only sources with at least 5 analyst-reviewed updates are included."
    )

    if df_feedback.empty:
        st.warning("No feedback data available for the selected period.")
    elif df_sources.empty:
        st.warning(
            "Master Sources sheet could not be loaded. "
            "Check that the sheet is publicly accessible (File → Share → Anyone with link → Viewer)."
        )
    else:
        with st.spinner("Computing source-level accuracy..."):
            reliable, poor = compute_source_accuracy(df_feedback, df_sources)

        if reliable.empty:
            st.warning(
                "Not enough reviewed data to compute source-level accuracy. "
                "At least 5 analyst-reviewed updates per source are required."
            )
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("##### ✅ Top 10 Most Reliably Scored Sources")
                st.caption(
                    "Sources where the model's score most consistently matched the analyst's score. "
                    "**Overall Agreement %** = % of reviewed updates where model score = analyst score. "
                    "These are sources the model handles well — likely consistent, clearly structured content."
                )
                reliable_display = reliable[[
                    c for c in [
                        "Jurisdiction","Document / Source","Industry","Authority","Banding",
                        "Analyst Reviewed","Overall Agreement %","Accuracy 3s %","Accuracy 1s %"
                    ] if c in reliable.columns
                ]]
                st.dataframe(
                    reliable_display.style.format({
                        "Overall Agreement %": "{:.1f}%",
                        "Accuracy 3s %": "{:.1f}%",
                        "Accuracy 1s %": "{:.1f}%",
                    }, na_rep="-"),
                    use_container_width=True,
                )
                if not reliable.empty:
                    st.bar_chart(
                        reliable.set_index("Jurisdiction")["Overall Agreement %"].head(10)
                    )

            with col_b:
                st.markdown("##### ❌ Top 10 Most Poorly Scored Sources")
                st.caption(
                    "Sources where the model's score most often differed from the analyst's score. "
                    "Low agreement % = the model is frequently getting these wrong. "
                    "These sources may need prompt tuning, re-classification, or manual review priority."
                )
                poor_display = poor[[
                    c for c in [
                        "Jurisdiction","Document / Source","Industry","Authority","Banding",
                        "Analyst Reviewed","Overall Agreement %","Accuracy 3s %","Accuracy 1s %"
                    ] if c in poor.columns
                ]]
                st.dataframe(
                    poor_display.style.format({
                        "Overall Agreement %": "{:.1f}%",
                        "Accuracy 3s %": "{:.1f}%",
                        "Accuracy 1s %": "{:.1f}%",
                    }, na_rep="-"),
                    use_container_width=True,
                )
                if not poor.empty:
                    st.bar_chart(
                        poor.set_index("Jurisdiction")["Overall Agreement %"].head(10)
                    )

            st.divider()
            st.markdown("##### 📋 Full Source Accuracy Table")
            st.caption(
                "All sources with at least 5 reviewed updates, sorted by agreement % descending. "
                "Use this to find any source you want to investigate."
            )
            all_stats, _ = compute_source_accuracy(df_feedback, df_sources)
            if not all_stats.empty:
                full_display = all_stats[[
                    c for c in [
                        "Jurisdiction","Document / Source","Industry","Authority","Banding",
                        "Total Updates","Analyst Reviewed","Overall Agreement %",
                        "Accuracy 3s %","Accuracy 1s %"
                    ] if c in all_stats.columns
                ]]
                st.dataframe(
                    full_display.style.format({
                        "Overall Agreement %": "{:.1f}%",
                        "Accuracy 3s %": "{:.1f}%",
                        "Accuracy 1s %": "{:.1f}%",
                    }, na_rep="-"),
                    use_container_width=True,
                )
