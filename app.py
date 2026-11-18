import streamlit as st

st.set_page_config(page_title="GambleRisk Guard", page_icon="🛡️", layout="wide")

st.title("GambleRisk Guard")
st.caption("Behavioral risk profiling for online betting accounts")

st.markdown(
    """
Choose a module from the sidebar:

- **Single Profile Simulator**: check one account's behavioral segment and risk level.
- **Batch CSV Auditor**: label many accounts at once and download the results.
- **Analytics Dashboard**: explore segments, risk distribution, and PCA projections.
"""
)

st.info(
    "Results are behavioral indications for analyst review, not a diagnosis of any individual.",
    icon="ℹ️",
)
