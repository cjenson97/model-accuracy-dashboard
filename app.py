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

def filter_analyst_reviewed(df):
    """Keep only rows where an analyst has reviewed the AI score (agreed or changed).
    Excludes: unscored (no ai_score) and not-yet-reviewed (no analyst_score)."""
    df = df.copy()
    df["ai_score"] = pd.to_numeric(df["ai_score"], errors="coerce")
    df["analyst_score"] = pd.to_numeric(df["analyst_score"], errors="coerce")
    return df[
        df["ai_score"].notna() &
        df["analyst_score"].notna()
    ]

def compute_score3_downgrade_breakdown(df):
    results = []
    for jurisdiction, grp in df.groupby("jurisdiction"):
        s3 = grp[grp["ai_score"] == 3]
        total = len(s3)
        if total == 0:
            continue
        agreed    = len(s3[s3["analyst_score"] == 3])
        to_2      = len(s3[s3["analyst_score"] == 2])
        to_1      = len(s3[s3["analyst_score"] == 1])
        results.append({
            "Jurisdiction":       jurisdiction,
            "AI Score 3 Total":   total,
            "Analyst Agreed (→3)": agreed,
            "Downgraded to 2":    to_2,
            "Downgraded to 1":    to_1,
            "% Agreed":           round(agreed / total * 100, 1),
            "% → 2":              round(to_2   / total * 100, 1),
            "% → 1":              round(to_1   / total * 100, 1),
        })
    return pd.DataFrame(results).sort_values("% Agreed") if results else pd.DataFrame()


def generate_interpretation(summary_df, downgrade_df, period_label):
    if summary_df.empty:
        return "No data available for interpretation."

    lines = [f"### Interpretation — {period_label}"]

    acc3 = summary_df["Accuracy 3s %"].dropna()
    avg3 = round(acc3.mean(), 1) if len(acc3) > 0 else None
    below_benchmark = summary_df[summary_df["Accuracy 3s %"].notna() & (summary_df["Accuracy 3s %"] < BENCHMARK_3S)]
    pct_below = round(len(below_benchmark) / len(acc3) * 100) if len(acc3) > 0 else 0

    if avg3 is not None:
        if avg3 < 50:
            severity = "critical — the model is agreeing with analysts on fewer than half of its Score 3 decisions"
        elif avg3 < BENCHMARK_3S:
            severity = f"below target — the model needs improvement to reach the {BENCHMARK_3S}% benchmark"
        else:
            severity = "on target"
        lines.append(f"**Overall Score 3 accuracy is {severity}.** Average across jurisdictions: {avg3}%.")

    if pct_below > 75:
        lines.append(
            f"**{pct_below}% of jurisdictions are below the {BENCHMARK_3S}% benchmark.** "
            "This is a systemic pattern, not isolated to one or two jurisdictions — the model is consistently over-scoring across the board."
        )
    elif pct_below > 40:
        lines.append(
            f"**{pct_below}% of jurisdictions are below benchmark.** "
            "This is widespread but not universal, suggesting the model struggles in specific contexts rather than having a fundamental calibration problem."
        )

    if not downgrade_df.empty:
        mostly_to_1 = downgrade_df[downgrade_df["% → 1"] > downgrade_df["% → 2"]]
        mostly_to_2 = downgrade_df[downgrade_df["% → 2"] >= downgrade_df["% → 1"]]

        if len(mostly_to_1) > len(mostly_to_2):
            lines.append(
                "**Analysts are mostly downgrading Score 3s all the way to Score 1** (not 2). "
                "This suggests the model is fundamentally misidentifying low-relevance content as highly relevant — "
                "a calibration or training data issue rather than a borderline scoring problem."
            )
        elif len(mostly_to_2) > 0:
            lines.append(
                "**Analysts are mostly downgrading Score 3s to Score 2** (not Score 1). "
                "This suggests borderline cases where the model is slightly over-scoring rather than making large errors — "
                "a threshold adjustment may be more appropriate than retraining."
            )

    above_benchmark = summary_df[summary_df["Accuracy 3s %"].notna() & (summary_df["Accuracy 3s %"] >= BENCHMARK_3S)]
    if not above_benchmark.empty:
        good = ", ".join(above_benchmark.sort_values("Accuracy 3s %", ascending=False)["Industry"].tolist())
        lines.append(
            f"**Jurisdictions meeting benchmark:** {good}. "
            "The model performs well here — reviewing what these jurisdictions have in common (data volume, document types, vertical) "
            "may reveal what conditions favour better model performance."
        )

    lines.append(
        "**Recommended next steps:** (1) Check if analyst scoring guidelines have changed recently. "
        "(2) Review the downgrade breakdown below to see if the pattern is consistent across jurisdictions. "
        "(3) Use the Accuracy Trends tab to check whether this is getting better or worse over time."
    )

    return "  \n\n".join(lines)


def compute_summary(df_all, df_reviewed):
    """
    df_all     — all Tier 1 feedback records (includes unreviewed)
    df_reviewed — only analyst-reviewed records (both ai_score and analyst_score present)

    Accuracy = accepted / total_all (not just reviewed) so unreviewed records
    reduce the accuracy figure rather than being excluded from it.
    """
    df_all = df_all.copy()
    df_all["ai_score"] = pd.to_numeric(df_all["ai_score"], errors="coerce")
    df_all["analyst_score"] = pd.to_numeric(df_all["analyst_score"], errors="coerce")

    results = []
    for industry, grp_all in df_all.groupby("jurisdiction"):
        grp_rev = df_reviewed[df_reviewed["jurisdiction"] == industry]

        total_3  = len(grp_all[grp_all["ai_score"] == 3])
        total_2  = len(grp_all[grp_all["ai_score"] == 2])
        total_1  = len(grp_all[grp_all["ai_score"] == 1])
        unscored = int(grp_all["ai_score"].isna().sum())

        accepted_3 = len(grp_rev[(grp_rev["ai_score"] == 3) & (grp_rev["analyst_score"] == 3)])
        accepted_2 = len(grp_rev[(grp_rev["ai_score"] == 2) & (grp_rev["analyst_score"] == 2)])
        accepted_1 = len(grp_rev[(grp_rev["ai_score"] == 1) & (grp_rev["analyst_score"] == 1)])

        # Denominator = ALL records for that score (including unreviewed)
        acc_3 = round(accepted_3 / total_3 * 100, 1) if total_3 > 0 else None
        acc_2 = round(accepted_2 / total_2 * 100, 1) if total_2 > 0 else None
        acc_1 = round(accepted_1 / total_1 * 100, 1) if total_1 > 0 else None

        pending = len(grp_all[
            (grp_all["ai_score"].isin([2, 3])) &
            (grp_all["analyst_score"].isna())
        ])

        results.append({
            "Industry":        industry,
            "Score 3":         total_3,
            "Score 2":         total_2,
            "Score 1":         total_1,
            "Unscored":        unscored,
            "3s Accepted":     accepted_3,
            "2s Accepted":     accepted_2,
            "1s Accepted":     accepted_1,
            "Accuracy 3s %":   acc_3,
            "Accuracy 2s %":   acc_2,
            "Accuracy 1s %":   acc_1,
            "Pending Review":  pending,
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
    """Data-only summary — no inference or narrative, only computed facts."""
    if summary_df.empty:
        return "No data available for this period."

    acc3 = summary_df["Accuracy 3s %"].dropna()
    acc1 = summary_df["Accuracy 1s %"].dropna()
    avg3 = round(acc3.mean(), 1) if len(acc3) > 0 else None
    avg1 = round(acc1.mean(), 1) if len(acc1) > 0 else None

    below3 = summary_df[
        summary_df["Accuracy 3s %"].notna() &
        (summary_df["Accuracy 3s %"] < BENCHMARK_3S)
    ].sort_values("Accuracy 3s %")

    below1 = summary_df[
        summary_df["Accuracy 1s %"].notna() &
        (summary_df["Accuracy 1s %"] < BENCHMARK_1S)
    ].sort_values("Accuracy 1s %")

    total_all     = int(summary_df[["Score 3", "Score 2", "Score 1"]].sum().sum())
    total_accepted = int(summary_df[["3s Accepted", "2s Accepted", "1s Accepted"]].sum().sum())
    total_pending  = int(summary_df["Pending Review"].sum())

    lines = [
        f"**Period:** {period_label}  |  "
        f"**Total scored records:** {total_all:,}  |  "
        f"**Analyst accepted:** {total_accepted:,}"
    ]

    if avg3 is not None:
        status3 = "✅ above" if avg3 >= BENCHMARK_3S else "🔴 below"
        lines.append(
            f"**Score 3 avg accuracy:** {avg3}% — {status3} the {BENCHMARK_3S}% benchmark "
            f"({len(acc3)} jurisdiction(s) with data)."
        )
        if len(below3) > 0:
            worst3_list = ", ".join(
                f"{row['Industry']} ({row['Accuracy 3s %']}%)"
                for _, row in below3.head(3).iterrows()
            )
            lines.append(
                f"**{len(below3)} jurisdiction(s) below Score 3 benchmark:** {worst3_list}"
                + (" ..." if len(below3) > 3 else ".")
            )
        else:
            lines.append("**All jurisdictions** are at or above the Score 3 benchmark.")

    if avg1 is not None:
        status1 = "✅ above" if avg1 >= BENCHMARK_1S else "🔴 below"
        lines.append(
            f"**Score 1 avg accuracy:** {avg1}% — {status1} the {BENCHMARK_1S}% benchmark "
            f"({len(acc1)} jurisdiction(s) with data)."
        )
        if len(below1) > 0:
            worst1_list = ", ".join(
                f"{row['Industry']} ({row['Accuracy 1s %']}%)"
                for _, row in below1.head(3).iterrows()
            )
            lines.append(
                f"**{len(below1)} jurisdiction(s) below Score 1 benchmark:** {worst1_list}"
                + (" ..." if len(below1) > 3 else ".")
            )
        else:
            lines.append("**All jurisdictions** are at or above the Score 1 benchmark.")

    if total_pending > 0:
        lines.append(
            f"**Pending review:** {total_pending:,} Score 2/3 updates not yet reviewed — "
            "accuracy figures will update as analysts complete reviews."
        )

    return "  \n".join(lines)

def generate_trend_summary(df_w, period_label):
    """Data-only trend summary — lists exact figures, no narrative inference."""
    if df_w.empty:
        return "No data available for the selected jurisdictions and period."

    acc3 = df_w.groupby("Industry")["Accuracy 3s %"].mean().dropna().round(1)
    acc1 = df_w.groupby("Industry")["Accuracy 1s %"].mean().dropna().round(1)

    below3 = acc3[acc3 < BENCHMARK_3S].sort_values()
    above3 = acc3[acc3 >= BENCHMARK_3S].sort_values(ascending=False)

    # Detect declining trend (last week worse than first week)
    declining = []
    for ind in df_w["Industry"].dropna().unique():
        sub = df_w[df_w["Industry"] == ind].sort_values("Week")["Accuracy 3s %"].dropna()
        if len(sub) >= 2 and sub.iloc[-1] < sub.iloc[0]:
            diff = round(sub.iloc[-1] - sub.iloc[0], 1)
            declining.append(f"{ind} ({diff:+.1f}%)")

    lines = [f"**Period:** {period_label}  |  **Jurisdictions shown:** {df_w['Industry'].nunique()}"]

    if len(acc3) > 0:
        lines.append(f"**Score 3 avg accuracy (period mean):** {round(acc3.mean(), 1)}%")

        if len(below3) > 0:
            below_list = ", ".join(f"{jx} ({v}%)" for jx, v in below3.head(5).items())
            lines.append(
                f"**{len(below3)} jurisdiction(s) averaged below {BENCHMARK_3S}% on Score 3:** {below_list}"
                + (" ..." if len(below3) > 5 else ".")
            )
        else:
            lines.append(f"**All selected jurisdictions** averaged at or above the {BENCHMARK_3S}% Score 3 benchmark.")

        if declining:
            lines.append(
                f"**Declining Score 3 trend (week-on-week):** {', '.join(declining[:5])}"
                + (" ..." if len(declining) > 5 else ".")
            )
        else:
            lines.append("**No declining Score 3 trends** detected across the selected period.")

    if len(acc1) > 0:
        lines.append(f"**Score 1 avg accuracy (period mean):** {round(acc1.mean(), 1)}%")

    return "  \n".join(lines)

def generate_sources_summary(df_top, period_label):
    """Data-only sources summary — exact counts and figures only."""
    if df_top.empty:
        return "No qualifying sources found."

    total    = len(df_top)
    below    = df_top[df_top["Overall Agreement %"].notna() & (df_top["Overall Agreement %"] < BENCHMARK_3S)]
    above    = df_top[df_top["Overall Agreement %"].notna() & (df_top["Overall Agreement %"] >= BENCHMARK_3S)]
    avg      = round(df_top["Overall Agreement %"].dropna().mean(), 1)
    worst_3  = df_top.head(3)[["Document / Source", "Jurisdiction", "Overall Agreement %"]].copy()

    lines = [
        f"**Period:** {period_label}  |  **Tier 1 sources assessed:** {total} (min. 5 analyst-reviewed records each)",
        f"**Average overall agreement:** {avg}%",
        f"**Below 80% agreement:** {len(below)} source(s) ({round(len(below)/total*100, 1)}%)",
        f"**At or above 80% agreement:** {len(above)} source(s) ({round(len(above)/total*100, 1)}%)",
    ]

    if not worst_3.empty:
        worst_list = "  \n".join(
            f"- {row['Document / Source']} — {row['Jurisdiction']} ({row['Overall Agreement %']}%)"
            for _, row in worst_3.iterrows()
        )
        lines.append(f"**3 lowest-agreement sources:**  \n{worst_list}")

    return "  \n".join(lines)

def compute_top_inaccurate_sources_6m(df):
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["ai_score"] = pd.to_numeric(df["ai_score"], errors="coerce")
    df["analyst_score"] = pd.to_numeric(df["analyst_score"], errors="coerce")

    # All analyst-reviewed records (agreed + changed)
    df = df[
        df["ai_score"].notna() &
        df["analyst_score"].notna()
    ]

    # Only Tier 1 jurisdictions
    df = filter_tier1(df)

    rows = []
    for (doc, jx), grp in df.groupby(["document_name", "jurisdiction"], dropna=False):
        reviewed = int(grp["analyst_score"].notna().sum())
        if reviewed < 5:
            continue

        total      = len(grp)
        s3         = int(len(grp[grp["ai_score"] == 3]))
        s2         = int(len(grp[grp["ai_score"] == 2]))
        s1         = int(len(grp[grp["ai_score"] == 1]))
        unscored   = int(grp["ai_score"].isna().sum())
        agreed     = int(len(grp[grp["ai_score"] == grp["analyst_score"]]))
        accepted_3 = int(len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)]))
        accepted_1 = int(len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)]))
        vertical   = grp["vertical"].mode()[0] if "vertical" in grp.columns and grp["vertical"].notna().any() else None

        rows.append({
            "Document / Source":     doc,
            "Jurisdiction":          jx,
            "Vertical":              vertical,
            "Analyst Reviewed":      reviewed,
            "AI Score 3 Count":      s3,
            "AI Score 2 Count":      s2,
            "AI Score 1 Count":      s1,
            "Overall Agreement %":   round(agreed / reviewed * 100, 1) if reviewed > 0 else None,
            "Accuracy 3s %":         round(accepted_3 / s3 * 100, 1) if s3 > 0 else None,
            "Accuracy 1s %":         round(accepted_1 / s1 * 100, 1) if s1 > 0 else None,
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
st.caption(
    "Showing **Tier 1 jurisdictions only** · "
    "Accuracy = % of AI scores an analyst agreed with (no change made) out of all analyst-reviewed records"
)

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
        df_scored_raw   = load_scored_updates(year, month_str)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if week_option != "All weeks":
    week_idx   = weeks.index(week_option)
    week_start = first_day + timedelta(weeks=week_idx)
    week_end   = min(week_start + timedelta(days=6), last_day)
    df_feedback_raw = filter_by_week(df_feedback_raw, "score_updated_date", week_start, week_end)
    df_scored_raw   = filter_by_week(df_scored_raw, "processed_time", week_start, week_end)

# Apply Tier 1 filter
df_feedback = filter_tier1(df_feedback_raw)
df_scored   = filter_tier1(df_scored_raw)

# All analyst-reviewed records (agreed + changed)
df_feedback_reviewed = filter_analyst_reviewed(df_feedback)

# Analyst-changed only — used for the Tab 4 "disagreements" view
df_feedback_changed = df_feedback_reviewed[
    df_feedback_reviewed["ai_score"] != df_feedback_reviewed["analyst_score"]
]

period_label = f"week of {week_option}" if week_option != "All weeks" else f"{month_name} {year}"

st.caption(
    f"**{len(df_feedback):,}** Tier 1 feedback rows "
    f"(**{len(df_feedback_reviewed):,}** analyst-reviewed · "
    f"**{len(df_feedback_changed):,}** analyst-changed) | "
    f"**{len(df_scored):,}** scored rows — {period_label}"
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Weekly Summary",
    "Accuracy Trends",
    "Scored Updates",
    "Top Inaccurate Sources (6M)",
    "Unreviewed Updates",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Weekly Summary
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model Accuracy Summary — Tier 1 Jurisdictions")
    st.caption(
        "Accuracy = % of AI scores the analyst agreed with (left unchanged). "
        "Includes all analyst-reviewed records. Unscored and un-reviewed records are excluded."
    )

    if df_feedback_reviewed.empty:
        st.warning("No analyst-reviewed records found for this period in Tier 1 jurisdictions.")
    else:
        summary = compute_summary(df_feedback, df_feedback_reviewed)

        with st.expander("Summary Insight", expanded=True):
            st.info(generate_summary_text(summary, period_label))

        styled = (
            summary.style
            .map(lambda v: colour_cell(v, BENCHMARK_3S), subset=["Accuracy 3s %"])
            .map(lambda v: colour_cell(v, BENCHMARK_1S), subset=["Accuracy 1s %"])
            .map(colour_score, subset=["Accuracy 2s %"])
            .format({
                "Accuracy 3s %": "{:.1f}%",
                "Accuracy 2s %": "{:.1f}%",
                "Accuracy 1s %": "{:.1f}%",
            }, na_rep="-")
        )
        st.dataframe(styled, use_container_width=True)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Score Distribution by Jurisdiction")
            st.caption("Number of analyst-reviewed records per AI score, by jurisdiction.")
            st.bar_chart(summary.set_index("Industry")[["Score 3", "Score 2", "Score 1"]])

        with col_b:
            st.markdown("##### Accuracy % by Jurisdiction")
            st.caption(f"Benchmarks: Score 3 = {BENCHMARK_3S}%, Score 1 = {BENCHMARK_1S}%. Denominator = all records (including unreviewed).")
            acc = (
                summary.set_index("Industry")[["Accuracy 3s %", "Accuracy 2s %", "Accuracy 1s %"]]
                .dropna(how="all")
            )
            st.bar_chart(acc)

        st.divider()

        st.markdown("##### Pending Analyst Review")
        st.caption("Tier 1 updates the AI scored 2 or 3 that haven't been reviewed yet.")
        st.bar_chart(summary.set_index("Industry")[["Pending Review"]])

        st.divider()

        # ── Score 3 Downgrade Breakdown ──────────────────────────────────────
        st.markdown("##### Score 3 Downgrade Breakdown")
        st.caption(
            "For every AI Score 3 record reviewed by an analyst: did they agree (→3), "
            "downgrade slightly (→2), or downgrade fully (→1)? "
            "This helps identify whether the model is slightly over-scoring or fundamentally miscalibrated."
        )

        downgrade_df = compute_score3_downgrade_breakdown(df_feedback_reviewed)

        if downgrade_df.empty:
            st.info("No Score 3 records with analyst reviews found for this period.")
        else:
            fmt_dg = {
                "% Agreed": "{:.1f}%",
                "% → 2":    "{:.1f}%",
                "% → 1":    "{:.1f}%",
            }

            def colour_agreed(val):
                if pd.isna(val):
                    return ""
                if val >= BENCHMARK_3S:
                    return "color: green; font-weight: bold"
                if val >= 50:
                    return "color: orange; font-weight: bold"
                return "color: red; font-weight: bold"

            styled_dg = (
                downgrade_df.style
                .map(colour_agreed, subset=["% Agreed"])
                .format(fmt_dg, na_rep="-")
            )
            st.dataframe(styled_dg, use_container_width=True)

            st.markdown("###### Where are Score 3s going?")
            chart_dg = (
                downgrade_df.set_index("Jurisdiction")[["% Agreed", "% → 2", "% → 1"]]
                .sort_values("% Agreed")
            )
            st.bar_chart(chart_dg)

        st.divider()

        # ── Interpretation ───────────────────────────────────────────────────
        with st.expander("Diagnostic Interpretation", expanded=True):
            interp = generate_interpretation(summary, downgrade_df if not downgrade_df.empty else pd.DataFrame(), period_label)
            st.warning(interp)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Weekly Accuracy Trend
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Weekly Accuracy Trend — Tier 1 Jurisdictions")
    st.caption(
        "Week-by-week accuracy based on all analyst-reviewed records (agreed + changed). "
        "Accuracy = % of AI scores the analyst left unchanged."
    )

    if df_feedback_reviewed.empty:
        st.warning("No analyst-reviewed records found for this period.")
    else:
        df_feedback_reviewed = df_feedback_reviewed.copy()
        df_feedback_reviewed["week"] = (
            pd.to_datetime(df_feedback_reviewed["score_updated_date"], errors="coerce")
            .dt.tz_localize(None)
            .dt.to_period("W")
            .astype(str)
        )

        weekly = []
        for (week, ind), grp in df_feedback_reviewed.groupby(["week", "jurisdiction"]):
            t3 = len(grp[grp["ai_score"] == 3])
            t1 = len(grp[grp["ai_score"] == 1])
            a3 = len(grp[(grp["ai_score"] == 3) & (grp["analyst_score"] == 3)])
            a1 = len(grp[(grp["ai_score"] == 1) & (grp["analyst_score"] == 1)])
            weekly.append({
                "Week":           week,
                "Industry":       ind,
                "Accuracy 3s %":  round(a3 / t3 * 100, 1) if t3 > 0 else None,
                "Accuracy 1s %":  round(a1 / t1 * 100, 1) if t1 > 0 else None,
            })

        df_w = pd.DataFrame(weekly)

        if df_w.empty:
            st.warning("Not enough data to build weekly trends.")
        else:
            industries = sorted(df_w["Industry"].dropna().unique())
            sel = st.multiselect("Filter by jurisdiction", industries, default=industries[:6])
            df_w_filtered = df_w[df_w["Industry"].isin(sel)]

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

        score_counts = df_scored["score"].value_counts().sort_index()
        total_sc = len(df_scored)
        pct3 = round(score_counts.get(3, 0) / total_sc * 100, 1) if total_sc > 0 else 0
        pct2 = round(score_counts.get(2, 0) / total_sc * 100, 1) if total_sc > 0 else 0
        pct1 = round(score_counts.get(1, 0) / total_sc * 100, 1) if total_sc > 0 else 0

        with st.expander("Score Distribution Insight", expanded=True):
            st.info(
                f"**Total updates processed:** {total_sc:,}  \n"
                f"**Score 3 (high relevance):** {score_counts.get(3, 0):,} ({pct3}%)  \n"
                f"**Score 2 (medium relevance):** {score_counts.get(2, 0):,} ({pct2}%)  \n"
                f"**Score 1 (low relevance):** {score_counts.get(1, 0):,} ({pct1}%)"
            )

        st.divider()

        t1_jx   = ["All"] + sorted(df_scored["jurisdiction"].dropna().unique())
        sel_ind = st.selectbox("Filter by Jurisdiction", t1_jx)
        view    = df_scored if sel_ind == "All" else df_scored[df_scored["jurisdiction"] == sel_ind]

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
        "Tier 1 jurisdictions only. Includes all analyst-reviewed records (agreed + changed). "
        "Sources with fewer than 5 reviewed records are excluded."
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
            t1_count      = filter_tier1(df_6m)
            reviewed_count = filter_analyst_reviewed(t1_count)
            changed_count  = reviewed_count[reviewed_count["ai_score"] != reviewed_count["analyst_score"]]

            st.caption(
                f"**{len(df_6m):,}** total records loaded · "
                f"**{len(t1_count):,}** in Tier 1 jurisdictions · "
                f"**{len(reviewed_count):,}** analyst-reviewed · "
                f"**{len(changed_count):,}** analyst-changed"
            )

            with st.spinner("Computing source accuracy..."):
                df_top = compute_top_inaccurate_sources_6m(df_6m)

            if df_top.empty:
                st.info("No Tier 1 sources met the minimum threshold (5+ analyst-reviewed records).")
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
                    "Accuracy 3s %":       "{:.1f}%",
                    "Accuracy 1s %":       "{:.1f}%",
                }
                acc_cols   = ["Overall Agreement %", "Accuracy 3s %", "Accuracy 1s %"]
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Unreviewed Updates
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Unreviewed Updates — Tier 1 Jurisdictions")
    st.caption(
        "Updates the AI has scored but no analyst has yet reviewed. "
        "Covers all AI scores (1, 2, and 3) for the selected period."
    )

    # Build unreviewed dataset: ai_score present, analyst_score absent
    df_unreviewed = df_feedback.copy()
    df_unreviewed["ai_score"] = pd.to_numeric(df_unreviewed["ai_score"], errors="coerce")
    df_unreviewed["analyst_score"] = pd.to_numeric(df_unreviewed["analyst_score"], errors="coerce")
    df_unreviewed = df_unreviewed[
        df_unreviewed["ai_score"].notna() &
        df_unreviewed["analyst_score"].isna()
    ]

    if df_unreviewed.empty:
        st.success("No unreviewed updates found for this period in Tier 1 jurisdictions.")
    else:
        total_unrev = len(df_unreviewed)
        unrev_3 = int((df_unreviewed["ai_score"] == 3).sum())
        unrev_2 = int((df_unreviewed["ai_score"] == 2).sum())
        unrev_1 = int((df_unreviewed["ai_score"] == 1).sum())

        # ── Key metrics ──────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Unreviewed", f"{total_unrev:,}")
        m2.metric("Score 3 Unreviewed", f"{unrev_3:,}",
                  help="High-relevance updates not yet checked by an analyst")
        m3.metric("Score 2 Unreviewed", f"{unrev_2:,}",
                  help="Medium-relevance updates not yet checked by an analyst")
        m4.metric("Score 1 Unreviewed", f"{unrev_1:,}",
                  help="Low-relevance updates not yet checked by an analyst")

        st.divider()

        # ── By jurisdiction breakdown ─────────────────────────────────────────
        by_jx = (
            df_unreviewed.groupby("jurisdiction")["ai_score"]
            .value_counts()
            .unstack(fill_value=0)
            .rename(columns={3: "Score 3", 2: "Score 2", 1: "Score 1"})
        )
        for col in ["Score 3", "Score 2", "Score 1"]:
            if col not in by_jx.columns:
                by_jx[col] = 0
        by_jx = by_jx[["Score 3", "Score 2", "Score 1"]]
        by_jx["Total"] = by_jx.sum(axis=1)
        by_jx = by_jx.sort_values("Total", ascending=False)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Unreviewed by Jurisdiction")
            st.caption("Sorted by total unreviewed count.")
            st.bar_chart(by_jx[["Score 3", "Score 2", "Score 1"]])

        with col_b:
            st.markdown("##### Unreviewed Count Table")
            st.dataframe(by_jx.reset_index().rename(columns={"jurisdiction": "Jurisdiction"}),
                         use_container_width=True)

        st.divider()

        # ── Filters ───────────────────────────────────────────────────────────
        st.markdown("##### Unreviewed Records")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            jx_opts = ["All"] + sorted(df_unreviewed["jurisdiction"].dropna().unique())
            sel_jx = st.selectbox("Filter by Jurisdiction", jx_opts, key="unrev_jx")

        with fc2:
            score_opts = ["All", "3 — High", "2 — Medium", "1 — Low"]
            sel_score = st.selectbox("Filter by AI Score", score_opts, key="unrev_score")

        with fc3:
            if "vertical" in df_unreviewed.columns:
                vt_opts = ["All"] + sorted(df_unreviewed["vertical"].dropna().unique())
                sel_vt = st.selectbox("Filter by Vertical", vt_opts, key="unrev_vt")
            else:
                sel_vt = "All"

        view_unrev = df_unreviewed.copy()
        if sel_jx != "All":
            view_unrev = view_unrev[view_unrev["jurisdiction"] == sel_jx]
        if sel_score != "All":
            score_val = int(sel_score[0])
            view_unrev = view_unrev[view_unrev["ai_score"] == score_val]
        if sel_vt != "All" and "vertical" in view_unrev.columns:
            view_unrev = view_unrev[view_unrev["vertical"] == sel_vt]

        # Select relevant display columns (handle optional columns gracefully)
        display_cols = [c for c in [
            "document_name", "jurisdiction", "vertical",
            "ai_score", "ai_score_reason",
            "created_date", "score_updated_date",
        ] if c in view_unrev.columns]

        st.caption(f"Showing **{len(view_unrev):,}** unreviewed records")
        st.dataframe(
            view_unrev[display_cols]
            .sort_values("score_updated_date", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )
