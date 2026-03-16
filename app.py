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

def colour_score(val):
    """Red for low accuracy, green for high — used in the cross-reference table."""
    if pd.isna(val) or val is None:
        return ""
    if val >= 80:
        return "color: green; font-weight: bold"
    if val >= 60:
        return "color: orange; font-weight: bold"
    return "color: red; font-weight: bold"

def compute_source_cross_reference(df_feedback, df_scored, df_sources):
    """
    Use the Master Sources 'Documents and URLs' tab as the reference list of sources.

    For each URL in the master sources sheet, look up how many updates in both
    feedback_loop_prod and scored_updates were scored 3 / 2 / 1 / unscored,
    and compute accuracy (analyst agreement) per score level.

    The join key is: master sources Jurisdiction <-> feedback jurisdiction,
    and master sources URL / document name <-> feedback document_name.

    Returns a single DataFrame ranked by most problematic sources (worst accuracy
    and highest unscored rate), plus a full table.
    """
    if df_sources.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ── Detect key columns in master sources ─────────────────────────────────
    url_col  = next((c for c in df_sources.columns if c.strip().upper() == "URL"), None)
    jx_col   = next((c for c in df_sources.columns if "jurisd"  in c.lower()), None)
    auth_col = next((c for c in df_sources.columns if "author"  in c.lower()), None)
    band_col = next((c for c in df_sources.columns if "band"    in c.lower()), None)
    ind_col  = next((c for c in df_sources.columns if "industr" in c.lower()), None)

    if not url_col or not jx_col:
        return pd.DataFrame(), pd.DataFrame()

    # Build slim master sources lookup: one row per URL
    keep   = [url_col, jx_col]
    rename = {url_col: "URL", jx_col: "Jurisdiction"}
    for col, name in [(auth_col, "Authority"), (band_col, "Banding"), (ind_col, "Industry")]:
        if col:
            keep.append(col)
            rename[col] = name

    src = (
        df_sources[keep]
        .rename(columns=rename)
        .dropna(subset=["URL"])
        .drop_duplicates(subset=["URL"])
        .copy()
    )
    src["_jx_key"] = src["Jurisdiction"].astype(str).str.strip().str.lower()

    # ── Build per-URL stats from feedback_loop_prod ───────────────────────────
    rows = []

    for _, src_row in src.iterrows():
        url = src_row["URL"]
        jx  = src_row["_jx_key"]

        # Match feedback rows for this jurisdiction
        fb_jx = df_feedback[
            df_feedback["jurisdiction"].astype(str).str.strip().str.lower() == jx
        ] if not df_feedback.empty else pd.DataFrame()

        # Match scored_updates rows for this jurisdiction
        sc_jx = df_scored[
            df_scored["jurisdiction"].astype(str).str.strip().str.lower() == jx
        ] if not df_scored.empty else pd.DataFrame()

        # Also try to narrow by document_name containing part of the URL hostname
        # (best-effort — falls back to all rows for that jurisdiction)
        if not fb_jx.empty and "document_name" in fb_jx.columns:
            try:
                from urllib.parse import urlparse
                host = urlparse(str(url)).netloc.replace("www.", "")
                if host:
                    fb_url = fb_jx[
                        fb_jx["document_name"].astype(str).str.contains(host, case=False, na=False)
                    ]
                    if not fb_url.empty:
                        fb_jx = fb_url
            except Exception:
                pass

        if not sc_jx.empty and "document_title" in sc_jx.columns:
            try:
                from urllib.parse import urlparse
                host = urlparse(str(url)).netloc.replace("www.", "")
                if host:
                    sc_url = sc_jx[
                        sc_jx["document_title"].astype(str).str.contains(host, case=False, na=False)
                    ]
                    if not sc_url.empty:
                        sc_jx = sc_url
            except Exception:
                pass

        total_fb   = len(fb_jx)
        total_sc   = len(sc_jx)
        total      = total_fb  # primary source for scoring stats

        if total == 0 and total_sc == 0:
            continue  # skip URLs with no data at all

        # Score counts from feedback
        s3 = int(len(fb_jx[fb_jx["ai_score"] == 3]))   if total_fb > 0 else 0
        s2 = int(len(fb_jx[fb_jx["ai_score"] == 2]))   if total_fb > 0 else 0
        s1 = int(len(fb_jx[fb_jx["ai_score"] == 1]))   if total_fb > 0 else 0
        su = int(len(fb_jx[fb_jx["ai_score"].isna()])) if total_fb > 0 else 0

        # Score counts from scored_updates (to capture all scored items, not just reviewed)
        sc_s3 = int(len(sc_jx[sc_jx["score"] == 3]))   if total_sc > 0 else 0
        sc_s2 = int(len(sc_jx[sc_jx["score"] == 2]))   if total_sc > 0 else 0
        sc_s1 = int(len(sc_jx[sc_jx["score"] == 1]))   if total_sc > 0 else 0
        sc_su = int(len(sc_jx[sc_jx["score"].isna()])) if total_sc > 0 else 0

        # Accuracy from feedback (analyst reviewed)
        reviewed   = int(fb_jx["analyst_score"].notna().sum()) if total_fb > 0 else 0
        agreed     = int(len(fb_jx[fb_jx["ai_score"] == fb_jx["analyst_score"]])) if total_fb > 0 else 0
        accepted_3 = int(len(fb_jx[(fb_jx["ai_score"] == 3) & (fb_jx["analyst_score"] == 3)])) if total_fb > 0 else 0
        accepted_1 = int(len(fb_jx[(fb_jx["ai_score"] == 1) & (fb_jx["analyst_score"] == 1)])) if total_fb > 0 else 0
        t2_rev     = int(len(fb_jx[(fb_jx["ai_score"] == 2) & (fb_jx["analyst_score"].notna())])) if total_fb > 0 else 0
        accepted_2 = int(len(fb_jx[(fb_jx["ai_score"] == 2) & (fb_jx["analyst_score"] == 2)])) if total_fb > 0 else 0

        row = {
            "URL":                  url,
            "Jurisdiction":         src_row.get("Jurisdiction", ""),
            "Industry":             src_row.get("Industry", pd.NA),
            "Authority":            src_row.get("Authority", pd.NA),
            "Banding":              src_row.get("Banding", pd.NA),
            # Feedback stats
            "Feedback Updates":     total_fb,
            "Analyst Reviewed":     reviewed,
            "Score 3 (fb)":         s3,
            "Score 2 (fb)":         s2,
            "Score 1 (fb)":         s1,
            "Unscored (fb)":        su,
            # Scored updates stats
            "Scored Updates":       total_sc,
            "Score 3 (scored)":     sc_s3,
            "Score 2 (scored)":     sc_s2,
            "Score 1 (scored)":     sc_s1,
            "Unscored (scored)":    sc_su,
            # Accuracy
            "Overall Agreement %":  round(agreed / reviewed * 100, 1) if reviewed > 0 else None,
            "Accuracy 3s %":        round(accepted_3 / s3 * 100, 1) if s3 > 0 else None,
            "Accuracy 2s %":        round(accepted_2 / t2_rev * 100, 1) if t2_rev > 0 else None,
            "Accuracy 1s %":        round((s1 - accepted_1) / s1 * 100, 1) if s1 > 0 else None,
            "Unscored Rate %":      round((su + sc_su) / (total_fb + total_sc) * 100, 1)
                                    if (total_fb + total_sc) > 0 else None,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.DataFrame(rows)

    # Only include sources that have any activity
    df_active = df_all[(df_all["Feedback Updates"] > 0) | (df_all["Scored Updates"] > 0)].copy()

    # Problematic = reviewed sources with low agreement, OR high unscored rate
    df_reviewed = df_active[df_active["Analyst Reviewed"] >= 3].copy()

    problematic = df_reviewed.sort_values(
        ["Overall Agreement %", "Unscored Rate %"],
        ascending=[True, False],
        na_position="last",
    )

    return problematic, df_active.sort_values(
        ["Overall Agreement %", "Unscored Rate %"],
        ascending=[True, False],
        na_position="last",
    )

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
            "὾2 Green = at or above benchmark. ὓ4 Red = below benchmark."
        )
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
            st.markdown("##### U0001f3af Model Accuracy % by Industry")
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
                st.markdown("##### U0001f4c8 Score 3 Accuracy % per Week")
                st.caption(
                    f"Each line = one industry over time. "
                    f"Benchmark line = {BENCHMARK_3S}%. "
                    "Below benchmark = model over-scoring updates as high relevance."
                )
                p3 = df_w.pivot(index="Week", columns="Industry", values="Accuracy 3s %")
                p3["── Benchmark"] = BENCHMARK_3S
                st.line_chart(p3)

            with col_b:
                st.markdown("##### U0001f4c9 Score 1 Accuracy % per Week")
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
            st.markdown("##### U0001f4ca Score Distribution")
            st.caption(
                "How the model distributed scores. Score 3 = high relevance, "
                "2 = medium, 1 = low."
            )
            st.bar_chart(df_scored["score"].value_counts().sort_index())

        with col_b:
            st.markdown("##### U0001f512 Confidence Score Distribution")
            st.caption(
                "How confident the model was in each score. "
                "Low confidence alongside incorrect scores may indicate model drift."
            )
            st.bar_chart(df_scored["confidence_score"].value_counts().sort_index())

        st.divider()

        industries = ["All"] + sorted(df_scored["jurisdiction"].dropna().unique())
        sel_ind = st.selectbox("Filter table by Industry", industries)
        view = df_scored if sel_ind == "All" else df_scored[df_scored["jurisdiction"] == sel_ind]

        st.markdown(f"##### U0001f5c2 Raw Records ({len(view):,} updates)")
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
    st.subheader("U0001f500 Source Cross-Reference: Problematic URLs by Score")
    st.caption(
        "Uses the **Master Sources list** (Documents and URLs tab) as the reference list of sources. "
        "For each URL, compares how the model scored updates from that source — "
        "showing Score 3 / Score 2 / Score 1 / Unscored counts from both "
        "**feedback_loop_prod** (with analyst accuracy) and **scored_updates** (full volume). "
        "Ranked by worst analyst agreement first, so the most problematic sources are at the top."
    )

    if df_sources.empty:
        st.warning(
            "Master Sources sheet could not be loaded. "
            "Make sure the sheet has been shared with your service account email "
            "and that the `gcp_service_account` credentials are set in Streamlit secrets."
        )
    else:
        with st.spinner("Cross-referencing master sources against scored data..."):
            df_problematic, df_all_active = compute_source_cross_reference(
                df_feedback, df_scored, df_sources
            )

        if df_problematic.empty and df_all_active.empty:
            st.info(
                "No matching data found between the master sources list and the scored data "
                "for this period. This may mean the jurisdiction names do not match, "
                "or there is no activity for the listed sources this month."
            )
        else:
            fmt = {
                "Overall Agreement %": "{:.1f}%",
                "Accuracy 3s %":       "{:.1f}%",
                "Accuracy 2s %":       "{:.1f}%",
                "Accuracy 1s %":       "{:.1f}%",
                "Unscored Rate %":     "{:.1f}%",
            }

            acc_cols = [c for c in ["Overall Agreement %","Accuracy 3s %","Accuracy 2s %","Accuracy 1s %"] if c in (df_problematic.columns if not df_problematic.empty else df_all_active.columns)]

            # ── Summary metrics ───────────────────────────────────────────────
            total_sources = len(df_all_active)
            reviewed_sources = len(df_problematic)
            col1, col2, col3 = st.columns(3)
            col1.metric("Sources in Master List with Activity", total_sources)
            col2.metric("Sources with Analyst Reviews (>=3)", reviewed_sources)
            if not df_problematic.empty and "Overall Agreement %" in df_problematic.columns:
                avg_agreement = round(df_problematic["Overall Agreement %"].mean(), 1)
                col3.metric("Avg Agreement % (reviewed sources)", f"{avg_agreement}%")

            st.divider()

            # ── Most problematic sources table ────────────────────────────────
            st.markdown("##### ❌ Most Problematic Sources — Worst Accuracy First")
            st.caption(
                "Each row is a URL from the Master Sources list. "
                "**Feedback Updates** = updates in feedback_loop_prod for this source's jurisdiction. "
                "**Scored Updates** = updates in scored_updates. "
                "**Score 3/2/1/Unscored (fb)** = breakdown from feedback data. "
                "**Score 3/2/1/Unscored (scored)** = breakdown from scored_updates. "
                "**Accuracy %** columns = how often the analyst agreed with the model for each score level. "
                "Sorted worst overall agreement first."
            )

            if not df_problematic.empty:
                prob_cols = [c for c in [
                    "URL", "Jurisdiction", "Industry", "Authority", "Banding",
                    "Feedback Updates", "Analyst Reviewed",
                    "Score 3 (fb)", "Score 2 (fb)", "Score 1 (fb)", "Unscored (fb)",
                    "Scored Updates",
                    "Score 3 (scored)", "Score 2 (scored)", "Score 1 (scored)", "Unscored (scored)",
                    "Overall Agreement %", "Accuracy 3s %", "Accuracy 2s %", "Accuracy 1s %",
                    "Unscored Rate %",
                ] if c in df_problematic.columns]

                styled_prob = df_problematic[prob_cols].style.format(fmt, na_rep="-")
                for col in acc_cols:
                    if col in prob_cols:
                        styled_prob = styled_prob.map(colour_score, subset=[col])

                st.dataframe(styled_prob, use_container_width=True)

                # Bar chart of overall agreement
                if "Overall Agreement %" in df_problematic.columns and "URL" in df_problematic.columns:
                    chart_df = (
                        df_problematic[["URL", "Overall Agreement %"]]
                        .dropna(subset=["Overall Agreement %"])
                        .head(20)
                        .set_index("URL")
                    )
                    st.bar_chart(chart_df)

            st.divider()

            # ── All active sources (including unreviewed) ─────────────────────
            st.markdown("##### U0001f4cb All Active Sources — Score & Unscored Counts")
            st.caption(
                "All URLs from the master sources list that had any activity this period, "
                "including those not yet reviewed by an analyst. "
                "Use this to spot sources with high unscored rates or unusual score distributions."
            )

            if not df_all_active.empty:
                all_cols = [c for c in [
                    "URL", "Jurisdiction", "Industry", "Authority", "Banding",
                    "Feedback Updates", "Analyst Reviewed",
                    "Score 3 (fb)", "Score 2 (fb)", "Score 1 (fb)", "Unscored (fb)",
                    "Scored Updates",
                    "Score 3 (scored)", "Score 2 (scored)", "Score 1 (scored)", "Unscored (scored)",
                    "Unscored Rate %",
                    "Overall Agreement %", "Accuracy 3s %", "Accuracy 2s %", "Accuracy 1s %",
                ] if c in df_all_active.columns]
                

                st.dataframe(
                    df_all_active[all_cols].style.format(fmt, na_rep="-"),
                    use_container_width=True,
                )
