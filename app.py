import streamlit as st
import pandas as pd
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
from datetime import datetime, timedelta

st.set_page_config(page_title="Model Accuracy Dashboard", layout="wide")

BENCHMARK_3S = 80.0
BENCHMARK_1S = 70.0

# ── Tier 1 jurisdictions (FS banding = 1. High from the banding sheet) ────────
TIER1_JURISDICTIONS = {
    "argentina", "australia", "austria", "belgium", "brazil", "bulgaria",
    "california", "chile", "croatia", "cyprus", "czech republic", "denmark",
    "estonia", "european union", "federal (canada)", "federal (us)", "finland",
    "florida", "france", "germany", "greece", "hong kong", "hungary", "india",
    "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta",
    "massachusetts", "netherlands", "new york", "new zealand", "norway",
    "ontario", "poland", "portugal", "romania", "singapore", "slovakia",
    "slovenia", "south africa", "south korea", "spain", "sweden", "switzerland",
    "turkey", "ukraine", "united kingdom", "united states, texas",
    "united states, virginia", "wales",
}

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
def load_feedback_loop_6m():
    today = datetime.today()
    unique_ym = []
    seen = set()
    d = today.replace(day=1)
    for _ in range(6):
        d = (d - timedelta(days=1)).replace(day=1)
        ym = (str(d.year), str(d.month).zfill(2))
        if ym not in seen:
            seen.add(ym)
            unique_ym.append(ym)
    conditions = " OR ".join(
        [f"(year = '{y}' AND month = '{m}')" for y, m in unique_ym]
    )
    return get_cursor().execute(f"""
        SELECT
            content_id, ai_meta_id, score_updated_date, created_date,
            document_name, jurisdiction, vertical,
            ai_score, ai_score_reason,
            analyst_score, analyst_score_reason,
            status, status_id
        FROM scans.feedback_loop_prod
        WHERE {conditions}
    """).as_pandas()

# ── Helpers ───────────────────────────────────────────────────────────────────
def filter_by_week(df, date_col, week_start, week_end):
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    return df[
        (dates >= pd.Timestamp(week_start).tz_localize(None)) &
        (dates <= pd.Timestamp(week_end).tz_localize(None))
    ]

def is_tier1(jurisdiction):
    return str(jurisdiction).strip().lower() in TIER1_JURISDICTIONS

def filter_tier1(df, jx_col="jurisdiction"):
    return df[df[jx_col].apply(is_tier1)]

def filter_analyst_changed(df):
    """Keep only rows where analyst actively disagreed with the AI score.
    Excludes: unscored (no ai_score), not reviewed (no analyst_score),
    and cases where analyst agreed."""
    df = df.copy()
    df["ai_score"] = pd.to_numeric(df["ai_score"], errors="coerce")
    df["analyst_score"] = pd.to_numeric(df["analyst_score"], errors="coerce")
    return df[
        df["ai_score"].notna() &
        df["analyst_score"].notna() &
        (df["ai_score"] != df["analyst_score"])
    ]

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
        pending = len(grp[(grp["ai_score"].isin([2, 3])) & (grp["analyst_score"].isna())])
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
            "Pending Review": pending,
        })
    return pd.DataFrame(results)

def colour_cell(val, benchmark):
    if pd.isna(val) or val is None:
        return ""
    return f"color: {'green' if val >= benchmark else 'red'}; font-weight: bold"

def colour_score(val):
    if pd.isna(val) or val is None:
        return ""
    if val >= 80:
        return "color: green; font-weight: bold"
    if val >= 60:
        return "color: orange; font-weight: bold"
    return "color: red; font-weight: bold"

def generate_summary_text(summary_df, period_label):
    """Generate a plain-English summary of the accuracy table."""
    if summary_df.empty:
        return "No data available for this period."
    acc3 = summary_df["Accuracy 3s %"].dropna()
    acc1 = summary_df["Accuracy 1s %"].dropna()
    below3 = summary_df[summary_df["Accuracy 3s %"].notna() & (summary_df["Accuracy 3s %"] < BENCHMARK_3S)]
    below1 = summary_df[summary_df["Accuracy 1s %"].notna() & (summary_df["Accuracy 1s %"] < BENCHMARK_1S)]
    avg3 = round(acc3.mean(), 1) if len(acc3) > 0 else None
    avg1 = round(acc1.mean(), 1) if len(acc1) > 0 else None
    total_pending = int(summary_df["Pending Review"].sum())
    lines = []
    if avg3 is not None:
        status3 = "above" if avg3 >= BENCHMARK_3S else "below"
        lines.append(f"For **{period_label}**, average Score 3 accuracy across Tier 1 jurisdictions is **{avg3}%** — {status3} the {BENCHMARK_3S}% benchmark.")
    if len(below3) > 0:
        worst3 = below3.sort_values("Accuracy 3s %").head(3)["Industry"].tolist()
        lines.append(f"**{len(below3)} jurisdiction(s)** are below the Score 3 benchmark. The worst performers are: **{', '.join(worst3)}**.")
    else:
        lines.append("All Tier 1 jurisdictions are meeting the Score 3 accuracy benchmark.")
    if avg1 is not None:
        status1 = "above" if avg1 >= BENCHMARK_1S else "below"
        lines.append(f"Average Score 1 accuracy is **{avg1}%** — {status1} the {BENCHMARK_1S}% benchmark.")
    if len(below1) > 0:
        worst1 = below1.sort_values("Accuracy 1s %").head(3)["Industry"].tolist()
        lines.append(f"**{len(below1)} jurisdiction(s)** are below the Score 1 benchmark, including: **{', '.join(worst1)}**.")
    if total_pending > 0:
        lines.append(f"There are **{total_pending:,} updates** still pending analyst review — accuracy figures may improve once reviewed.")
    return " ".join(lines)

def generate_trend_summary(df_w, period_label):
    """Generate a plain-English summary of the weekly trend data."""
    if df_w.empty:
        return "Not enough data to summarise trends."
    lines = []
    acc3 = df_w.groupby("Industry")["Accuracy 3s %"].mean().dropna()
    declining = []
    for ind in df_w["Industry"].dropna().unique():
        sub = df_w[df_w["Industry"] == ind].sort_values("Week")["Accuracy 3s %"].dropna()
        if len(sub) >= 2 and sub.iloc[-1] < sub.iloc[0]:
            declining.append(ind)
    below_avg = acc3[acc3 < BENCHMARK_3S]
    if len(below_avg) > 0:
        lines.append(f"Over the selected weeks, **{len(below_avg)} jurisdiction(s)** averaged below the {BENCHMARK_3S}% Score 3 benchmark: **{', '.join(below_avg.index.tolist()[:3])}**{'...' if len(below_avg) > 3 else ''}.")
    else:
        lines.append("All selected jurisdictions averaged above the Score 3 benchmark over the selected period.")
    if declining:
        lines.append(f"**Declining trend** in Score 3 accuracy detected for: **{', '.join(declining[:3])}**{'...' if len(declining) > 3 else ''}. These jurisdictions scored better earlier in the period.")
    return " ".join(lines) if lines else "Accuracy trends look stable across the selected period."

def generate_sources_summary(df_top, period_label):
    """Generate a plain-English summary of the top inaccurate sources."""
    if df_top.empty:
        return "No qualifying sources found."
    total = len(df_top)
    below = len(df_top[df_top["Overall Agreement %"].notna() & (df_top["Overall Agreement %"] < BENCHMARK_3S)])
    avg = round(df_top["Overall Agreement %"].dropna().mean(), 1)
    worst = df_top.head(3)["Document / Source"].tolist()
    lines = [
        f"Across the last 6 months, **{total} Tier 1 sources** had enough analyst-reviewed records to be assessed.",
        f"**{below} of those ({round(below/total*100)}%)** are below the 80% agreement threshold — meaning analysts disagreed with the AI on more than 1 in 5 updates.",
        f"Average analyst agreement across all reviewed Tier 1 sources is **{avg}%**.",
        f"The most problematic sources right now are: **{', '.join(worst)}**.",
    ]
    return " ".join(lines)

def compute_top_inaccurate_sources_6m(df):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ai_score"] = pd.to_numeric(df["ai_score"], errors="coerce")
    df["analyst_score"] = pd.to_numeric(df["analyst_score"], errors="coerce")
    # Only analyst-changed records
    df = df[
        df["ai_score"].notna() &
        df["analyst_score"].notna() &
        (df["ai_score"] != df["analyst_score"])
    ]
    # Only Tier 1 jurisdictions
    df = filter_tier1(df)
    rows = []
    for (doc, jx), grp in df.groupby(["document_name", "jurisdiction"], dropna=False):
        reviewed = int(grp["analyst_score"].notna().sum())
        if reviewed < 5:
            continue
        total = len(grp)
        s3 = int(len(grp[grp["ai_score"] == 3]))
        s2 = int(len(grp[grp["ai_score"] == 2]))
        s1 = int(len(grp[grp["ai_score"] == 1]))
        unscored = int(grp["ai_score"].isna().sum())
        agreed = int(len(grp[grp["ai_score"] == grp["analyst_score"]]))
        accepted_3 = int(len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)]))
        accepted_1 = int(len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)]))
        vertical = grp["vertical"].mode()[0] if "vertical" in grp.columns and grp["vertical"].notna().any() else None
        rows.append({
            "Document / Source": doc,
            "Jurisdiction": jx,
            "Vertical": vertical,
            "Total Disagreements": total,
            "Analyst Reviewed": reviewed,
            "AI Score 3 → Changed": s3,
            "AI Score 2 → Changed": s2,
            "AI Score 1 → Changed": s1,
            "Overall Agreement %": round(agreed / reviewed * 100, 1) if reviewed > 0 else None,
            "Accuracy 3s %": round(accepted_3 / s3 * 100, 1) if s3 > 0 else None,
            "Accuracy 1s %": round((s1 - accepted_1) / s1 * 100, 1) if s1 > 0 else None,
        })
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["Overall Agreement %", "Analyst Reviewed"], ascending=[True, False], na_position="last")
        .reset_index(drop=True)
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Model Accuracy Dashboard")
st.caption("Showing **Tier 1 jurisdictions only** · Accuracy based on records where an analyst actively changed the AI score")

# ── Period selector ───────────────────────────────────────────────────────────
year_col, month_col, week_col, period_col = st.columns(4)

month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

with year_col:
    year = st.selectbox(
        "Year",
        options=[str(y) for y in range(2023, datetime.today().year + 1)],
        index=datetime.today().year - 2023,
    )

with month_col:
    month_name = st.selectbox("Month", options=month_names, index=datetime.today().month - 1)
    month_str = str(month_names.index(month_name) + 1).zfill(2)

first_day = datetime(int(year), int(month_str), 1)
if first_day.month == 12:
    last_day = first_day.replace(year=first_day.year + 1, month=1, day=1) - timedelta(days=1)
else:
    last_day = first_day.replace(month=first_day.month + 1, day=1) - timedelta(days=1)

weeks = []
current = first_day
while current <= last_day:
    week_end_day = min(current + timedelta(days=6), last_day)
    weeks.append(f"{current.strftime('%d %b')} - {week_end_day.strftime('%d %b')}")
    current += timedelta(days=7)

with week_col:
    week_option = st.selectbox("Week", options=["All weeks"] + weeks)

with period_col:
    st.metric("Period", f"{month_name[:3]} {year}")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading data..."):
        df_feedback_raw = load_feedback_loop(year, month_str)
        df_scored_raw = load_scored_updates(year, month_str)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if week_option != "All weeks":
    week_idx = weeks.index(week_option)
    week_start = first_day + timedelta(weeks=week_idx)
    week_end = min(week_start + timedelta(days=6), last_day)
    df_feedback_raw = filter_by_week(df_feedback_raw, "score_updated_date", week_start, week_end)
    df_scored_raw = filter_by_week(df_scored_raw, "processed_time", week_start, week_end)

# Apply Tier 1 filter
df_feedback = filter_tier1(df_feedback_raw)
df_scored = filter_tier1(df_scored_raw)

# Analyst-changed only (for accuracy calculations)
df_feedback_changed = filter_analyst_changed(df_feedback)

period_label = f"week of {week_option}" if week_option != "All weeks" else f"{month_name} {year}"
st.caption(
    f"**{len(df_feedback):,}** Tier 1 feedback rows "
    f"(**{len(df_feedback_changed):,}** analyst-changed) | "
    f"**{len(df_scored):,}** scored rows — {period_label}"
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Weekly Summary",
    "Accuracy Trends",
    "Scored Updates",
    "Top Inaccurate Sources (6M)",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Weekly Summary
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model Accuracy Summary — Tier 1 Jurisdictions")
    st.caption(
        "Only records where an analyst reviewed **and actively changed** the AI score are included. "
        "Unscored updates and un-reviewed records are excluded from accuracy calculations."
    )

    if df_feedback_changed.empty:
        st.warning("No analyst-changed records found for this period in Tier 1 jurisdictions.")
    else:
        summary = compute_summary(df_feedback_changed)

        # ── AI-generated summary ──────────────────────────────────────────────
        with st.expander("Summary Insight", expanded=True):
            st.info(generate_summary_text(summary, period_label))

        styled = (
            summary.style
            .map(lambda v: colour_cell(v, BENCHMARK_3S), subset=["Accuracy 3s %"])
            .map(lambda v: colour_cell(v, BENCHMARK_1S), subset=["Accuracy 1s %"])
            .format({"Accuracy 3s %": "{:.1f}%", "Accuracy 1s %": "{:.1f}%"}, na_rep="-")
        )
        st.dataframe(styled, use_container_width=True)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Score Distribution by Industry")
            st.caption("How many updates the analyst changed per score level, by jurisdiction.")
            st.bar_chart(
                summary.set_index("Industry")[["Score 3", "Score 2", "Score 1"]]
            )
        with col_b:
            st.markdown("##### Accuracy % by Industry")
            st.caption(f"Benchmarks: 3s = {BENCHMARK_3S}%, 1s = {BENCHMARK_1S}%")
            acc = (
                summary.set_index("Industry")[["Accuracy 3s %", "Accuracy 1s %"]]
                .dropna(how="all")
            )
            st.bar_chart(acc)

        st.divider()
        st.markdown("##### Pending Analyst Review")
        st.caption("Tier 1 updates scored 2 or 3 that haven't been reviewed yet.")
        st.bar_chart(summary.set_index("Industry")[["Pending Review"]])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Weekly Accuracy Trend
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Weekly Accuracy Trend — Tier 1 Jurisdictions")
    st.caption("Week-by-week accuracy based on analyst-changed records only.")

    if df_feedback_changed.empty:
        st.warning("No analyst-changed records found for this period.")
    else:
        df_feedback_changed["week"] = (
            pd.to_datetime(df_feedback_changed["score_updated_date"], errors="coerce")
            .dt.tz_localize(None)
            .dt.to_period("W")
            .astype(str)
        )

        weekly = []
        for (week, ind), grp in df_feedback_changed.groupby(["week", "jurisdiction"]):
            t3 = len(grp[grp["ai_score"] == 3])
            t1 = len(grp[grp["ai_score"] == 1])
            a3 = len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)])
            a1 = len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)])
            weekly.append({
                "Week": week,
                "Industry": ind,
                "Accuracy 3s %": round(a3 / t3 * 100, 1) if t3 > 0 else None,
                "Accuracy 1s %": round((t1 - a1) / t1 * 100, 1) if t1 > 0 else None,
            })

        df_w = pd.DataFrame(weekly)

        if df_w.empty:
            st.warning("Not enough data to build weekly trends.")
        else:
            industries = sorted(df_w["Industry"].dropna().unique())
            sel = st.multiselect("Filter by jurisdiction", industries, default=industries[:6])
            df_w_filtered = df_w[df_w["Industry"].isin(sel)]

            # ── AI-generated summary ──────────────────────────────────────────
            with st.expander("Trend Insight", expanded=True):
                st.info(generate_trend_summary(df_w_filtered, period_label))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### Score 3 Accuracy % per Week")
                p3 = df_w_filtered.pivot(index="Week", columns="Industry", values="Accuracy 3s %")
                p3["-- Benchmark"] = BENCHMARK_3S
                st.line_chart(p3)
            with col_b:
                st.markdown("##### Score 1 Accuracy % per Week")
                p1 = df_w_filtered.pivot(index="Week", columns="Industry", values="Accuracy 1s %")
                p1["-- Benchmark"] = BENCHMARK_1S
                st.line_chart(p1)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Scored Updates Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Scored Updates — Tier 1 Jurisdictions")
    st.caption("Raw model output for Tier 1 jurisdictions. No analyst feedback included.")

    if df_scored.empty:
        st.warning("No scored_updates data for Tier 1 jurisdictions in the selected period.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Score Distribution")
            st.bar_chart(df_scored["score"].value_counts().sort_index())
        with col_b:
            st.markdown("##### Confidence Score Distribution")
            st.bar_chart(df_scored["confidence_score"].value_counts().sort_index())

        # Quick summary
        score_counts = df_scored["score"].value_counts().sort_index()
        total_sc = len(df_scored)
        pct3 = round(score_counts.get(3, 0) / total_sc * 100, 1) if total_sc > 0 else 0
        pct1 = round(score_counts.get(1, 0) / total_sc * 100, 1) if total_sc > 0 else 0
        with st.expander("Score Distribution Insight", expanded=True):
            st.info(
                f"The model processed **{total_sc:,} updates** from Tier 1 jurisdictions this period. "
                f"**{pct3}%** were rated high relevance (Score 3) and **{pct1}%** were rated low relevance (Score 1). "
                f"A high proportion of Score 3s means the model is flagging a lot as important — cross-reference with the accuracy tab to check if analysts agree."
            )

        st.divider()

        t1_jx = ["All"] + sorted(df_scored["jurisdiction"].dropna().unique())
        sel_ind = st.selectbox("Filter by Jurisdiction", t1_jx)
        view = df_scored if sel_ind == "All" else df_scored[df_scored["jurisdiction"] == sel_ind]

        st.markdown(f"##### Raw Records ({len(view):,} updates)")
        st.dataframe(
            view[[
                "content_id", "document_title", "jurisdiction", "vertical",
                "score", "confidence_score", "confidence_score_in_words",
                "llm_model", "processed_time",
            ]].sort_values("processed_time", ascending=False),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Top Inaccurate Sources (6M) — lazy loaded
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Top Inaccurate Sources — Tier 1, Last 6 Months")
    st.caption(
        "Tier 1 jurisdictions only. Only records where an analyst **actively changed** the AI score are counted. "
        "Sources with fewer than 5 such records are excluded."
    )

    if st.button("Load 6-Month Data", type="primary"):
        try:
            with st.spinner("Pulling last 6 months from Athena..."):
                df_6m = load_feedback_loop_6m()
        except Exception as e:
            st.error(f"Error loading 6-month data: {e}")
            df_6m = pd.DataFrame()

        if df_6m.empty:
            st.warning("No data returned for the last 6 months.")
        else:
            t1_count = filter_tier1(df_6m)
            changed_count = filter_analyst_changed(t1_count)
            st.caption(
                f"**{len(df_6m):,}** total records loaded · "
                f"**{len(t1_count):,}** in Tier 1 jurisdictions · "
                f"**{len(changed_count):,}** analyst-changed"
            )

            with st.spinner("Computing source accuracy..."):
                df_top = compute_top_inaccurate_sources_6m(df_6m)

            if df_top.empty:
                st.info("No Tier 1 sources met the minimum threshold (5+ analyst-changed records).")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Tier 1 Sources Analysed", len(df_top))
                below = len(df_top[
                    df_top["Overall Agreement %"].notna() &
                    (df_top["Overall Agreement %"] < BENCHMARK_3S)
                ])
                col2.metric("Sources Below 80% Agreement", below)
                avg = round(df_top["Overall Agreement %"].dropna().mean(), 1)
                col3.metric("Avg Overall Agreement %", f"{avg}%")

                # ── AI-generated summary ──────────────────────────────────────
                with st.expander("Sources Insight", expanded=True):
                    st.info(generate_sources_summary(df_top, "last 6 months"))

                st.divider()

                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    sel_jx = st.multiselect(
                        "Filter by Jurisdiction",
                        sorted(df_top["Jurisdiction"].dropna().unique()),
                    )
                with filter_col2:
                    sel_vt = st.multiselect(
                        "Filter by Vertical",
                        sorted(df_top["Vertical"].dropna().unique()),
                    )

                df_display = df_top.copy()
                if sel_jx:
                    df_display = df_display[df_display["Jurisdiction"].isin(sel_jx)]
                if sel_vt:
                    df_display = df_display[df_display["Vertical"].isin(sel_vt)]

                top_n = st.slider(
                    "Show top N worst sources",
                    min_value=5,
                    max_value=min(100, len(df_display)),
                    value=min(20, len(df_display)),
                )
                df_display = df_display.head(top_n)

                st.divider()

                fmt = {
                    "Overall Agreement %": "{:.1f}%",
                    "Accuracy 3s %": "{:.1f}%",
                    "Accuracy 1s %": "{:.1f}%",
                }
                acc_cols = ["Overall Agreement %", "Accuracy 3s %", "Accuracy 1s %"]
                styled_top = df_display.style.format(fmt, na_rep="-")
                for col in acc_cols:
                    styled_top = styled_top.map(colour_score, subset=[col])

                st.markdown(f"##### Worst {top_n} Tier 1 Sources by Analyst Agreement")
                st.dataframe(styled_top, use_container_width=True)

                st.divider()

                chart_df = (
                    df_display[["Document / Source", "Overall Agreement %"]]
                    .dropna(subset=["Overall Agreement %"])
                    .set_index("Document / Source")
                )
                st.markdown("##### Overall Agreement % — Worst Sources")
                st.bar_chart(chart_df)
    else:
        st.info("Click **Load 6-Month Data** above to run this analysis. It queries Athena and may take 20-40 seconds.")
