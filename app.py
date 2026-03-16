import streamlit as st
import pandas as pd
from pyathena import connect

# Pull credentials from Streamlit secrets
conn = connect(
    aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
    s3_staging_dir="s3://YOUR-RESULTS-BUCKET/athena-results/",  # <-- update this
    region_name="eu-west-2",
    work_group="primary",
)

st.title("Feedback Loop Dashboard")

@st.cache_data(ttl=600)  # Cache for 10 minutes for "live-ish" data
def load_data():
    query = """
        SELECT 
            content_id,
            ai_meta_id,
            score_updated_date,
            created_date,
            document_name,
            jurisdiction,
            ai_score
        FROM scans.feedback_loop_prod
        ORDER BY score_updated_date DESC
        LIMIT 1000
    """
    return pd.read_sql(query, conn)

df = load_data()

st.dataframe(df)
st.bar_chart(df["ai_score"])  # Example chart — customise as needed
