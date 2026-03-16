import streamlit as st
import pandas as pd
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

def get_cursor():
    return connect(
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
        s3_staging_dir=st.secrets["aws"]["S3_STAGING_DIR"],
        region_name="eu-west-2",
        work_group="primary",
        cursor_class=PandasCursor,
    ).cursor()

@st.cache_data(ttl=600)
def load_data():
    cursor = get_cursor()
    return cursor.execute("""
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
    """).as_pandas()

st.title("Feedback Loop Dashboard")

try:
    df = load_data()
    st.dataframe(df)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
```

And your `requirements.txt`:
```
streamlit
pyathena[pandas]
boto3
pandas
