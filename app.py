import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyathena import connect

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Model Accuracy Dashboard",
    layout="wide",
    page_icon="📊"
)

# ──────────────────────────────────────────
# ATHENA CONNECTION
# Pull credentials from Streamlit secrets
# ──────────────────────────────────────────
@st.cache_resource
def get_connection():
    return connect(
        s3_staging_dir=st.secrets["athena"]["s3_staging_dir"],
        region_name=st.secrets["athena"]["region"],
        schema_name=st.secrets["athena"]["database"],
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
    )

# ──────────────────────────────────────────
# DATA QUERIES
# scored_updates columns:
#   raw_content_id, content_id, document_title,
#   document, jurisdiction, authority, vertical, score, score_reason
#
# feedback_loop_prod columns:
#   content_id, ai_meta_id, score_updated_date,
#   created_date, document_name, jurisdiction,
#   document, ai_score, ai_score_reason, analyst
# ──────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_scored_updates():
    query = """
        SELECT
            raw_content_id,
            content_id,
            document_title,
            jurisdiction,
            authority,
            vertical,
            score,
            score_reason
        FROM scored_updates
        WHERE score IS NOT NULL
    """
    return pd.read_sql(query, get_connection())

@st.cache_data(ttl=3600)
def load_feedback_loop():
    query = """
        SELECT
            content_id,
            ai_meta_id,
            score_updated_date,
            created_date,
            document_name,
            jurisdiction,
            ai_score,
            ai_score_reason,
            analyst
        FROM feedback_loop_prod
        WHERE ai_score IS NOT NULL
    """
    return pd.read_sql(query, get_connection())


# ──────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────
with st.spinner("Loading data from Athena..."):
    scored_df = load_scored_updates()
    feedback_df = load_feedback_loop()

# Convert dates
feedback_df["score_updated_date"] = pd.to_datetime(
    feedback_df["score_updated_date"], errors="coerce"
)
feedback_df["created_date"] = pd.to_datetime(
    feedback_df["created_date"], errors="coerce"
)
feedback_df["week"] = feedback_df["score_updated_date"].dt.to_period("W").astype(str)


# ──────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────
st.sidebar.title("🔍 Filters")

all_jurisdictions = sorted(scored_df["jurisdiction"].dropna().unique())
all_verticals     = sorted(scored_df["vertical"].dropna().unique())
all_analysts      = sorted(feedback_df["analyst"].dropna().unique())

selected_jurisdictions = st.sidebar.multiselect(
    "Jurisdiction", all_jurisdictions, default=all_jurisdictions
)
selected_verticals = st.sidebar.multiselect(
    "Vertical / Industry", all_verticals, default=all_verticals
)
selected_analysts = st.sidebar.multiselect(
    "Analyst", all_analysts, default=all_analysts
)

# Apply filters
scored_filtered = scored_df[
    scored_df["jurisdiction"].isin(selected_jurisdictions) &
    scored_df["vertical"].isin(selected_verticals)
]
feedback_filtered = feedback_df[
    feedback_df["jurisdiction"].isin(selected_jurisdictions) &
    feedback_df["analyst"].isin(selected_analysts)
]


# ──────────────────────────────────────────
# BENCHMARKS
# Update these numbers to match your targets
# ──────────────────────────────────────────
BENCHMARK_3S = 85.0
BENCHMARK_1S = 90.0


# ──────────────────────────────────────────
# COMPUTED METRICS
# ──────────────────────────────────────────

# Score distribution from scored_updates
score_counts = scored_filtered.groupby(
    ["vertical", "score"]
)["content_id"].count().reset_index()
score_counts.columns = ["vertical", "score", "count"]

# Accuracy per vertical from feedback_loop
# 3s accuracy = ai_score==3 accepted (analyst==1) / total ai_score==3
# 1s accuracy = ai_score==1 NOT accepted / total ai_score==1
score3 = feedback_filtered[feedback_filtered["ai_score"] == 3].groupby("jurisdiction").agg(
    total_3s=("content_id", "count"),
    accepted_3s=("analyst", lambda x: (x == 1).sum())
).reset_index()
score3["accuracy_3s"] = (score3["accepted_3s"] / score3["total_3s"] * 100).round(1)

score1 = feedback_filtered[feedback_filtered["ai_score"] == 1].groupby("jurisdiction").agg(
    total_1s=("content_id", "count"),
    accepted_1s=("analyst", lambda x: (x == 1).sum())
).reset_index()
score1["accuracy_1s"] = (
    (score1["total_1s"] - score1["accepted_1s"]) / score1["total_1s"] * 100
).round(1)

accuracy_df = score3.merge(score1, on="jurisdiction", how="outer")

# Weekly trend
weekly = feedback_filtered.groupby(["week", "ai_score"])["content_id"].count().reset_index()
weekly.columns = ["week", "ai_score", "count"]

# Pending review (score 2 and 3 not yet reviewed — analyst is null)
pending = scored_filtered[
    scored_filtered["score"].isin([2, 3])
]["content_id"].count()

# Analyst breakdown
analyst_breakdown = feedback_filtered.groupby(
    ["analyst", "ai_score"]
)["content_id"].count().reset_index()
analyst_breakdown.columns = ["analyst", "ai_score", "count"]


# ──────────────────────────────────────────
# DASHBOARD LAYOUT
# ──────────────────────────────────────────
st.title("📊 Model Accuracy Dashboard")
st.caption("Data sourced from `scored_updates` and `feedback_loop_prod` via Amazon Athena")

# ── KPI ROW ──────────────────────────────
avg_3s  = accuracy_df["accuracy_3s"].mean()
avg_1s  = accuracy_df["accuracy_1s"].mean()
total_u = len(scored_filtered)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Avg 3s Accuracy",
    f"{avg_3s:.1f}%" if not pd.isna(avg_3s) else "N/A",
    delta=f"{avg_3s - BENCHMARK_3S:+.1f}% vs benchmark" if not pd.isna(avg_3s) else None,
    delta_color="normal" if not pd.isna(avg_3s) and avg_3s >= BENCHMARK_3S else "inverse"
)
k2.metric(
    "Avg 1s Accuracy",
    f"{avg_1s:.1f}%" if not pd.isna(avg_1s) else "N/A",
    delta=f"{avg_1s - BENCHMARK_1S:+.1f}% vs benchmark" if not pd.isna(avg_1s) else None,
    delta_color="normal" if not pd.isna(avg_1s) and avg_1s >= BENCHMARK_1S else "inverse"
)
k3.metric("Total Updates Scored", f"{total_u:,}")
k4.metric("Pending Review (Score 2s & 3s)", f"{pending:,}")

st.divider()


# ── SCORE DISTRIBUTION BY VERTICAL ───────
st.subheader("Score Distribution by Vertical / Industry")
fig = px.bar(
    score_counts, x="vertical", y="count", color="score",
    barmode="stack",
    color_discrete_map={1: "#ef4444", 2: "#f59e0b", 3: "#22c55e"},
    labels={"count": "Number of Updates", "vertical": "Vertical", "score": "Score"}
)
st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── ACCURACY VS BENCHMARK ─────────────────
st.subheader("Accuracy vs Benchmarks by Jurisdiction")
tab1, tab2 = st.tabs(["Score 3s Accuracy", "Score 1s Accuracy"])

with tab1:
    if not accuracy_df.empty:
        fig = px.bar(
            accuracy_df.sort_values("accuracy_3s", ascending=False),
            x="jurisdiction", y="accuracy_3s",
            color="accuracy_3s",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            labels={"accuracy_3s": "Accuracy (%)", "jurisdiction": "Jurisdiction"},
            range_color=[50, 100]
        )
        fig.add_hline(
            y=BENCHMARK_3S, line_dash="dash", line_color="red",
            annotation_text=f"Benchmark {BENCHMARK_3S}%"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not accuracy_df.empty:
        fig = px.bar(
            accuracy_df.sort_values("accuracy_1s", ascending=False),
            x="jurisdiction", y="accuracy_1s",
            color="accuracy_1s",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            labels={"accuracy_1s": "Accuracy (%)", "jurisdiction": "Jurisdiction"},
            range_color=[50, 100]
        )
        fig.add_hline(
            y=BENCHMARK_1S, line_dash="dash", line_color="red",
            annotation_text=f"Benchmark {BENCHMARK_1S}%"
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── WEEKLY TREND ──────────────────────────
st.subheader("Weekly Volume by Score")
fig = px.line(
    weekly, x="week", y="count", color="ai_score",
    markers=True,
    color_discrete_map={1: "#ef4444", 2: "#f59e0b", 3: "#22c55e"},
    labels={"count": "Number of Updates", "week": "Week", "ai_score": "Score"}
)
st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── BENCHMARK COMPARISON TABLE ────────────
st.subheader("Full Accuracy Table")
if not accuracy_df.empty:
    display_df = accuracy_df.copy()
    display_df["3s vs Benchmark"] = display_df["accuracy_3s"] - BENCHMARK_3S
    display_df["1s vs Benchmark"] = display_df["accuracy_1s"] - BENCHMARK_1S

    def colour_delta(val):
        if pd.isna(val):
            return ""
        return "color: green" if val >= 0 else "color: red"

    st.dataframe(
        display_df[[
            "jurisdiction", "total_3s", "accepted_3s", "accuracy_3s",
            "total_1s", "accepted_1s", "accuracy_1s",
            "3s vs Benchmark", "1s vs Benchmark"
        ]].style
            .applymap(colour_delta, subset=["3s vs Benchmark", "1s vs Benchmark"])
            .format({
                "accuracy_3s": "{:.1f}%",
                "accuracy_1s": "{:.1f}%",
                "3s vs Benchmark": "{:+.1f}%",
                "1s vs Benchmark": "{:+.1f}%"
            }, na_rep="N/A"),
        use_container_width=True
    )

st.divider()


# ── ANALYST BREAKDOWN ─────────────────────
st.subheader("Analyst Breakdown")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Updates by analyst and score**")
    fig = px.bar(
        analyst_breakdown, x="analyst", y="count", color="ai_score",
        barmode="stack",
        color_discrete_map={1: "#ef4444", 2: "#f59e0b", 3: "#22c55e"},
        labels={"count": "Updates", "analyst": "Analyst", "ai_score": "Score"}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Score reason breakdown**")
    reason_counts = feedback_filtered["ai_score_reason"].value_counts().reset_index()
    reason_counts.columns = ["reason", "count"]
    fig = px.pie(
        reason_counts, names="reason", values="count",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)


# ── RAW DATA EXPLORER ─────────────────────
st.divider()
st.subheader("🔎 Raw Data Explorer")
explore_tab1, explore_tab2 = st.tabs(["scored_updates", "feedback_loop_prod"])

with explore_tab1:
    st.dataframe(scored_filtered, use_container_width=True)

with explore_tab2:
    st.dataframe(feedback_filtered, use_container_width=True)
