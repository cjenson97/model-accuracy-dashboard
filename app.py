import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
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
    """Load Documents and URLs tab from Master Sources via authenticated service account.
    Uses get_all_values() to safely handle duplicate or blank column headers."""
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly",
            ],
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = sheet.worksheet("Documents and URLs")

        # Use get_all_values() to avoid crash on duplicate/blank headers
        all_values = worksheet.get_all_values()
        if not all_values:
            return pd.DataFrame()

        raw_headers = all_values[0]

        # Deduplicate and clean headers
        seen = {}
        clean_headers = []
        for h in raw_headers:
            h = h.strip()
            if not h:
                h = "_blank"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            clean_headers.append(h)

        df = pd.DataFrame(all_values[1:], columns=clean_headers)

        # Drop columns with blank / unnamed headers
        df = df.loc[:, ~df.columns.str.startswith("_blank")]
        df = df.replace("", pd.NA)

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
            "Industry":       industry,
            "Score 3":        total_3,
            "Score 2":        total_2,
            "Score 1":        total_1,
            "Unscored":       unscored,
            "3s Accepted":    accepted_3,
            "1s Accepted":    accepted_1,
            "Accuracy 3s %":  acc_3,
            "Accuracy 1s %":  acc_1,
            "Pending Review": pending,
        })
    return pd.DataFrame(results)

def colour_cell(val, benchmark):
    if pd.isna(val) or val is None:
        return ""
    return f"color: {'green' if val >= benchmark else 'red'}; font-weight: bold"

def compute_source_accuracy(df_feedback, df_sources):
    """
    Cross-reference feedback_loop_prod with the Master Sources sheet (Documents and URLs tab).

    Joins on document_name (feedback) <-> URL (master sources).
    Computes per-URL accuracy broken down by Score 3, Score 2, Score 1, and Unscored.
    Returns (worst_urls_top20, best_urls_top10, full_table).
    """
    if df_feedback.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Step 1: compute stats per jurisdiction + document_name ──────────────
    group_cols = (
        ["jurisdiction", "document_name"]
        if "document_name" in df_feedback.columns
        else ["jurisdiction"]
    )

    source_stats = []
    for keys, grp in df_feedback.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)

        total      = len(grp)
        reviewed   = int(grp["analyst_score"].notna().sum())
        total_3    = int(len(grp[grp["ai_score"] == 3]))
        total_2    = int(len(grp[grp["ai_score"] == 2]))
        total_1    = int(len(grp[grp["ai_score"] == 1]))
        unscored   = int(len(grp[grp["ai_score"].isna()]))
        accepted_3 = int(len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)]))
        accepted_1 = int(len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)]))
        agreed     = int(len(grp[grp["ai_score"] == grp["analyst_score"]]))

        # Accuracy 2s: analyst agreed the score was 2 (medium)
        total_2_rev = int(len(grp[(grp["ai_score"] == 2) & (grp["analyst_score"].notna())]))
        accepted_2  = int(len(grp[(grp["ai_score"] == 2) & (grp["analyst_score"] == 2)]))

        source_stats.append({
            "Jurisdiction":        keys[0],
            "Document / Source":   keys[1] if len(keys) > 1 else "-",
            "Total Updates":       total,
            "Analyst Reviewed":    reviewed,
            "Score 3 Count":       total_3,
            "Score 2 Count":       total_2,
            "Score 1 Count":       total_1,
            "Unscored Count":      unscored,
            "Overall Agreement %": round(agreed / reviewed * 100, 1) if reviewed > 0 else None,
            "Accuracy 3s %":       round(accepted_3 / total_3 * 100, 1) if total_3 > 0 else None,
            "Accuracy 2s %":       round(accepted_2 / total_2_rev * 100, 1) if total_2_rev > 0 else None,
            "Accuracy 1s %":       round((total_1 - accepted_1) / total_1 * 100, 1) if total_1 > 0 else None,
            "Unscored Rate %":     round(unscored / total * 100, 1) if total > 0 else None,
        })

    if not source_stats:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_stats = pd.DataFrame(source_stats)

    # ── Step 2: enrich with URL + metadata from master sources ───────────────
    if not df_sources.empty:
        # Detect column names dynamically
        url_col  = next((c for c in df_sources.columns if c.strip().upper() == "URL"), None)
        jx_col   = next((c for c in df_sources.columns if "jurisd"  in c.lower()), None)
        auth_col = next((c for c in df_sources.columns if "author"  in c.lower()), None)
        band_col = next((c for c in df_sources.columns if "band"    in c.lower()), None)
        ind_col  = next((c for c in df_sources.columns if "industr" in c.lower()), None)
        doc_col  = next((c for c in df_sources.columns
                         if any(k in c.lower() for k in ["document", "title", "name", "source"])
                         and c.strip().upper() != "URL"), None)

        if jx_col:
            keep   = [jx_col]
            rename = {jx_col: "Jurisdiction"}
            if url_col:
                keep.append(url_col)
                rename[url_col] = "URL"
            if doc_col and doc_col not in keep:
                keep.append(doc_col)
                rename[doc_col] = "Source Name"
            for col, name in [(auth_col, "Authority"), (band_col, "Banding"), (ind_col, "Industry")]:
                if col and col not in keep:
                    keep.append(col)
                    rename[col] = name

            src_slim = df_sources[keep].rename(columns=rename).copy()
            src_slim["_jx_key"] = src_slim["Jurisdiction"].astype(str).str.strip().str.lower()
            df_stats["_jx_key"] = df_stats["Jurisdiction"].astype(str).str.strip().str.lower()

            df_stats = df_stats.merge(
                src_slim.drop(columns=["Jurisdiction"]),
                on="_jx_key",
                how="left",
            ).drop(columns=["_jx_key"])

    # ── Step 3: filter to sources with at least 3 reviewed updates ───────────
    df_enough = df_stats[df_stats["Analyst Reviewed"] >= 3].copy()

    # Worst = lowest Overall Agreement %, then penalise high unscored rate
    worst = df_enough.sort_values(
        ["Overall Agreement %", "Unscored Rate %"],
        ascending=[True, False],
        na_position="last",
    ).head(20)

    best = df_enough.sort_values("Overall Agreement %", ascending=False).head(10)
    full = df_enough.sort_values("Overall Agreement %", ascending=True)

    return worst, best, full

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
    "Data refreshes every 10 minutes."
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
    week_idx    = weeks.index(week_option)
    week_start  = first_day + timedelta(weeks=week_idx)
    week_end    = min(week_start + timedelta(days=6), last_day)
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
            st.bar_chart(
                summary.set_index("Industry")[["Score 3","Score 2","Score 1","Unscored"]]
            )

        with col_b:
            st.markdown("##### 🎯 Model Accuracy % by Industry")
            st.caption(
                f"3s and 1s accuracy side by side. "
                f"Benchmarks: 3s = {BENCHMARK_3S}%, 1s = {BENCHMARK_1S}%. "
                "Industries below benchmark need closer attention."
            )
            acc = (
                summary.set_index("Industry")[["Accuracy 3s %","Accuracy 1s %"]]
                .dropna(how="all")
            )
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
                "Week":          week,
                "Industry":      ind,
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
                st.markdown("##### 📈 Score 3 Accuracy % per Week")
                st.caption(
                    f"Each line = one industry over time. "
                    f"Benchmark line = {BENCHMARK_3S}%. "
                    "Below benchmark = model over-scoring updates as high relevance."
                )
                p3 = df_w.pivot(index="Week", columns="Industry", values="Accuracy 3s %")
                p3["── Benchmark"] = BENCHMARK_3S
                st.line_chart(p3)

            with col_b:
                st.markdown("##### 📉 Score 1 Accuracy % per Week")
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
    st.subheader("🔀 Source Cross-Reference: Most Problematic URLs & Sources")
    st.caption(
        "Cross-references **feedback_loop_prod** with the **Master Sources sheet** "
        "(Documents and URLs tab). Joins on jurisdiction + document name to look up each source's URL, "
        "then ranks sources by how poorly the model scored them across all score levels "
        "(Score 3, Score 2, Score 1, and Unscored). "
        "Only sources with at least 3 analyst-reviewed updates are included."
    )

    if df_feedback.empty:
        st.warning("No feedback data available for the selected period.")
    elif df_sources.empty:
        st.warning(
            "Master Sources sheet could not be loaded. "
            "Make sure the sheet has been shared with your service account email "
            "and that the `gcp_service_account` credentials are set in Streamlit secrets."
        )
    else:
        with st.spinner("Computing source-level accuracy..."):
            worst, best, full = compute_source_accuracy(df_feedback, df_sources)

        if worst.empty and best.empty:
            st.warning(
                "Not enough reviewed data to compute source-level accuracy. "
                "At least 3 analyst-reviewed updates per source are required."
            )
        else:
            # Determine which columns exist after the merge
            possible_cols = [
                "URL", "Jurisdiction", "Document / Source", "Industry", "Authority", "Banding",
                "Analyst Reviewed", "Total Updates",
                "Overall Agreement %",
                "Score 3 Count", "Accuracy 3s %",
                "Score 2 Count", "Accuracy 2s %",
                "Score 1 Count", "Accuracy 1s %",
                "Unscored Count", "Unscored Rate %",
            ]
            ref_df = worst if not worst.empty else best
            display_cols = [c for c in possible_cols if c in ref_df.columns]

            fmt = {
                "Overall Agreement %": "{:.1f}%",
                "Accuracy 3s %":       "{:.1f}%",
                "Accuracy 2s %":       "{:.1f}%",
                "Accuracy 1s %":       "{:.1f}%",
                "Unscored Rate %":     "{:.1f}%",
            }

            # ── Worst sources ─────────────────────────────────────────────────
            st.markdown("##### ❌ Top 20 Most Problematic Sources (Lowest Agreement)")
            st.caption(
                "Sources where the model most often disagreed with the analyst — ranked worst first. "
                "**Overall Agreement %** = % of reviewed updates where model score = analyst score. "
                "Columns show volume and accuracy broken down by each score level, plus the unscored rate. "
                "These sources need urgent attention: prompt tuning, re-classification, or manual review priority."
            )

            if not worst.empty:
                worst_display = [c for c in display_cols if c in worst.columns]
                st.dataframe(
                    worst[worst_display].style.format(fmt, na_rep="-"),
                    use_container_width=True,
                )

                # Bar chart — overall agreement per source (use URL if available, else Document / Source)
                label_col = "URL" if "URL" in worst.columns else "Document / Source"
                if label_col in worst.columns and "Overall Agreement %" in worst.columns:
                    chart_data = (
                        worst[[label_col, "Overall Agreement %"]]
                        .dropna(subset=["Overall Agreement %"])
                        .set_index(label_col)
                    )
                    st.bar_chart(chart_data)
            else:
                st.info("No problematic sources found for this period.")

            st.divider()

            # ── Score-level breakdown ────────────────────────────────────────
            st.markdown("##### 📊 Accuracy Breakdown by Score Level (Worst Sources)")
            st.caption(
                "For the most problematic sources, this shows how accuracy varies across "
                "Score 3 (high relevance), Score 2 (medium), Score 1 (low relevance), and Unscored rate. "
                "Helps pinpoint whether the model struggles most with a particular score level."
            )

            if not worst.empty:
                score_breakdown_cols = [
                    c for c in [
                        "URL", "Document / Source", "Jurisdiction",
                        "Score 3 Count", "Accuracy 3s %",
                        "Score 2 Count", "Accuracy 2s %",
                        "Score 1 Count", "Accuracy 1s %",
                        "Unscored Count", "Unscored Rate %",
                    ] if c in worst.columns
                ]
                st.dataframe(
                    worst[score_breakdown_cols].style.format(fmt, na_rep="-"),
                    use_container_width=True,
                )

            st.divider()

            # ── Best sources ─────────────────────────────────────────────────
            st.markdown("##### ✅ Top 10 Most Reliably Scored Sources (Highest Agreement)")
            st.caption(
                "Sources where the model's score most consistently matched the analyst's. "
                "These are sources the model handles well — usually consistent, clearly structured content."
            )

            if not best.empty:
                best_display = [c for c in display_cols if c in best.columns]
                st.dataframe(
                    best[best_display].style.format(fmt, na_rep="-"),
                    use_container_width=True,
                )
                label_col = "URL" if "URL" in best.columns else "Document / Source"
                if label_col in best.columns and "Overall Agreement %" in best.columns:
                    st.bar_chart(
                        best[[label_col, "Overall Agreement %"]]
                        .dropna(subset=["Overall Agreement %"])
                        .set_index(label_col)
                    )
            else:
                st.info("No reliable sources found for this period.")

            st.divider()

            # ── Full table ───────────────────────────────────────────────────
            st.markdown("##### 📋 Full Source Accuracy Table (All Sources, Worst First)")
            st.caption(
                "All sources with at least 3 reviewed updates, sorted by agreement % ascending (worst first). "
                "Use this to investigate any specific source or URL."
            )

            if not full.empty:
                full_display = [c for c in display_cols if c in full.columns]
                st.dataframe(
                    full[full_display].style.format(fmt, na_rep="-"),
                    use_container_width=True,
                )
