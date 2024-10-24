import streamlit as st
import pandas as pd
from datetime import datetime as _dt
import ssl; ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(
    page_title="Check list", 
    page_icon="🗓"
)
st.logo('assets/bbsw_logo.png')
st.session_state = {}

# df = pd.read_csv(f'data/checkin/{_dt.now().date()}.csv')
df = pd.read_csv(f'data/checkin/template.csv')


st.dataframe(df, use_container_width=True)