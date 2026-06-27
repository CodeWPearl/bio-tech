"""Main Streamlit application for the Cancer Mutation Pathogenicity Predictor.

Run with::

    streamlit run webapp/app.py
"""

from __future__ import annotations

import streamlit as st

from webapp.pages import (
    about,
    batch_analysis,
    data_explorer,
    home,
    model_performance,
    single_prediction,
)
from webapp.utils.api_client import APIClient
from webapp.utils.styling import get_custom_css

st.set_page_config(
    page_title="Cancer Mutation Pathogenicity Predictor",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)

client = APIClient()

PAGES = {
    "\U0001f3e0 Home": home,
    "\U0001f52c Single Prediction": single_prediction,
    "\U0001f4ca Batch Analysis": batch_analysis,
    "\U0001f4c8 Model Performance": model_performance,
    "\U0001f9ea Data Explorer": data_explorer,
    "\U0001f4d6 About": about,
}

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0">
            <span style="font-size:2.5rem">\U0001f9ec</span>
            <h2 style="margin:0.3rem 0 0 0;font-size:1.1rem">
                Cancer Mutation<br>Pathogenicity Predictor
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")

    health = client.health_check()
    if health and health.get("status") == "healthy":
        st.markdown(
            '<span style="color:#28A745">&#9679; API Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#DC3545">&#9675; API Disconnected</span>',
            unsafe_allow_html=True,
        )

    st.caption(f"API: {client.api_url}")

page_module = PAGES[selected_page]
page_module.render(client)
