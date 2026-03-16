import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyathena import connect

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Model Accuracy Dashboard",
    layout="wide",
    page_icon="📊"
)

ATHENA_S3_STAGING = st.secrets["athena"]["s3_staging_dir"]
ATHENA_REGION     = st.secrets["athena"]["region"]
ATHENA_DATABASE   = st.secrets["athena"]["database"]

# Benchmarks (update these or wire to Sheets later)
BENCHMARK_3S = 85.0  # % target for 3s accuracy
BENCHMARK_1S = 90.0  # % target for 1s accuracy


# ──────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────
@st.cache_resource
def get_connection():
    return connect(
        s3_staging_dir=ATHENA_S3_STAGING,
        region_name=ATHENA_REGION,
        schema_name=ATHENA_DATABASE
    )

@st.cache_data(ttl=3600)
def load_scored_updates():
    query = """
        SELECT
            week,
            industry,
            analyst,
            jurisdiction,
            score,
            COUNT(*) AS total_updates,
            SUM(CASE WHEN accepted = true THEN 1 ELSE 0 END) AS accepted_updates
        FROM scored_updates
        GROUP BY week, industry, analyst, jurisdiction, score
    """
    return pd.read_sql(query, get_connection())

@st.cache_data(ttl=3600)
def load_feedback_loop():
    query = """
        SELECT
            week,
            industry,
            analyst,
            jurisdiction,
            published_count
        FROM feedback_loop_prod
    """
    return pd.read_sql(query, get_connection())


# ──────────────────────────────────────────
# COMPUTED METRICS
# ──────────────────────────────────────────
def compute_accuracy(df):
    """
    For score 3s: accuracy = accepted_3s / total_3s * 100
    For score 1s: accuracy = (total_1s - accepted_1s) / total_1s * 100
    """
    score3 = df[df["score"] == 3].groupby(["week", "industry"]).agg(
        total_3s=("total_updates", "sum"),
        accepted_3s=("accepted_updates", "sum")
    ).reset_index()
    score3["accuracy_3s"] = (score3["accepted_3s"] / score3["total_3s"] * 100).round(1)

    score1 = df[df["score"] == 1].groupby(["week", "industry"]).agg(
        total_1s=("total_updates", "sum"),
        accepted_1s=("accepted_updates", "sum")
    ).reset_index()
    score1["accuracy_1s"] = (
        (score1["total_1s"] - score1["accepted_1s"]) / score1["total_1s"] * 100
    ).round(1)

    return score3.merge(score1, on=["week", "industry"], how="outer")

def compute_pending_review(df):
    return df[df["score"].isin([2, 3])].groupby(["week", "industry"]).agg(
        pending=("total_updates", "sum")
    ).reset_index()


# ──────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────
st.sidebar.title("Filters")

raw_df    = load_scored_updates()
fl_df     = load_feedback_loop()

all_weeks      = sorted(raw_df["week"].unique(), reverse=True)
all_industries = sorted(raw_df["industry"].unique())
all_analysts   = sorted(raw_df["analyst"].unique())

selected_weeks = st.sidebar.select_slider(
    "Week range",
    options=all_weeks,
    value=(all_weeks[-1], all_weeks[0])
)
selected_industries = st.sidebar.multiselect(
    "Industries", all_industries, default=all_industries
)
selected_analysts = st.sidebar.multiselect(
    "Analysts", all_analysts, default=all_analysts
)

# Apply filters
mask = (
    raw_df["week"].between(*selected_weeks) &
    raw_df["industry"].isin(selected_industries) &
    raw_df["analyst"].isin(selected_analysts)
)
filtered_df = raw_df[mask]
filtered_fl = fl_df[
    fl_df["week"].between(*selected_weeks) &
    fl_df["industry"].isin(selected_industries) &
    fl_df["analyst"].isin(selected_analysts)
]


# ──────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────
st.title("📊 Model Accuracy Dashboard")

accuracy_df = compute_accuracy(filtered_df)
pending_df  = compute_pending_review(filtered_df)

# ── KPI ROW ──────────────────────────────
latest_week = accuracy_df["week"].max()
latest      = accuracy_df[accuracy_df["week"] == latest_week]

avg_3s  = latest["accuracy_3s"].mean()
avg_1s  = latest["accuracy_1s"].mean()
total_u = filtered_df[filtered_df["week"] == latest_week]["total_updates"].sum()
pending = pending_df[pending_df["week"] == latest_week]["pending"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg 3s Accuracy (latest week)", f"{avg_3s:.1f}%",
          delta=f"Benchmark: {BENCHMARK_3S}%",
          delta_color="normal" if avg_3s >= BENCHMARK_3S else "inverse")
k2.metric("Avg 1s Accuracy (latest week)", f"{avg_1s:.1f}%",
          delta=f"Benchmark: {BENCHMARK_1S}%",
          delta_color="normal" if avg_1s >= BENCHMARK_1S else "inverse")
k3.metric("Total Updates (latest week)", f"{int(total_u):,}")
k4.metric("Pending Review (3s + 2s)", f"{int(pending):,}")

st.divider()


# ── WEEKLY ACCURACY TRENDS ────────────────
st.subheader("Weekly Model Accuracy by Industry")
tab1, tab2 = st.tabs(["Score 3s Accuracy", "Score 1s Accuracy"])

with tab1:
    fig = px.line(
        accuracy_df, x="week", y="accuracy_3s", color="industry",
        markers=True, labels={"accuracy_3s": "Accuracy (%)", "week": "Week"}
    )
    fig.add_hline(y=BENCHMARK_3S, line_dash="dash", line_color="red",
                  annotation_text=f"Benchmark {BENCHMARK_3S}%")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.line(
        accuracy_df, x="week", y="accuracy_1s", color="industry",
        markers=True, labels={"accuracy_1s": "Accuracy (%)", "week": "Week"}
    )
    fig.add_hline(y=BENCHMARK_1S, line_dash="dash", line_color="red",
                  annotation_text=f"Benchmark {BENCHMARK_1S}%")
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── SCORE DISTRIBUTION ────────────────────
st.subheader("Score Distribution by Industry")
score_dist = filtered_df.groupby(["industry", "score"])["total_updates"].sum().reset_index()
fig = px.bar(
    score_dist, x="industry", y="total_updates", color="score",
    barmode="stack", labels={"total_updates": "Updates", "score": "Score"},
    color_discrete_map={1: "#ef4444", 2: "#f59e0b", 3: "#22c55e"}
)
st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── BENCHMARK COMPARISON TABLE ────────────
st.subheader("Performance vs Benchmarks")

bm_table = accuracy_df[accuracy_df["week"] == latest_week][
    ["industry", "accuracy_3s", "accuracy_1s"]
].copy()
bm_table["3s vs Benchmark"] = bm_table["accuracy_3s"] - BENCHMARK_3S
bm_table["1s vs Benchmark"] = bm_table["accuracy_1s"] - BENCHMARK_1S

def colour_delta(val):
    colour = "green" if val >= 0 else "red"
    return f"color: {colour}"

st.dataframe(
    bm_table.style
        .applymap(colour_delta, subset=["3s vs Benchmark", "1s vs Benchmark"])
        .format({
            "accuracy_3s": "{:.1f}%",
            "accuracy_1s": "{:.1f}%",
            "3s vs Benchmark": "{:+.1f}%",
            "1s vs Benchmark": "{:+.1f}%"
        }),
    use_container_width=True
)

st.divider()


# ── ANALYST DRILLDOWN ─────────────────────
st.subheader("Analyst-Level Breakdown")

analyst_scores = filtered_df.groupby(["analyst", "score"])["total_updates"].sum().reset_index()
analyst_pub    = filtered_fl.groupby("analyst")["published_count"].sum().reset_index()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Updates received by score**")
    fig = px.bar(
        analyst_scores, x="analyst", y="total_updates", color="score",
        barmode="stack", labels={"total_updates": "Updates"},
        color_discrete_map={1: "#ef4444", 2: "#f59e0b", 3: "#22c55e"}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Updates published**")
    fig = px.bar(
        analyst_pub, x="analyst", y="published_count",
        labels={"published_count": "Published Updates"},
        color_discrete_sequence=["#6366f1"]
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── PENDING REVIEW ────────────────────────
st.subheader("Pending Review Queue (Score 3s & 2s)")
fig = px.bar(
    pending_df, x="week", y="pending", color="industry",
    barmode="stack", labels={"pending": "Updates Pending Review"}
)
st.plotly_chart(fig, use_container_width=True)
