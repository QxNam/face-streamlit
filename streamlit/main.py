import streamlit as st
from st_pages import hide_pages

st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout="wide"
)
st.logo('assets/bbsw_logo.png')
st.title("BBSW")


hide_pages(["main"])