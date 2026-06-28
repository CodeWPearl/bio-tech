"""Main Streamlit application for the Cancer Mutation Pathogenicity Predictor.

Run with::

    streamlit run webapp/app.py
"""

from __future__ import annotations

import streamlit as st

from webapp.views import (
    about,
    api_docs,
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
    "\U0001f3e0  Home": home,
    "\U0001f52c  Single Prediction": single_prediction,
    "\U0001f4ca  Batch Analysis": batch_analysis,
    "\U0001f4c8  Model Performance": model_performance,
    "\U0001f9ea  Data Explorer": data_explorer,
    "\U0001f4bb  API Docs": api_docs,
    "\U0001f4d6  About": about,
}

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1.5rem 0 1rem 0">
            <div style="
                width:60px;height:60px;margin:0 auto 0.8rem;
                background:linear-gradient(135deg,#6366F1,#8B5CF6);
                border-radius:16px;display:flex;align-items:center;
                justify-content:center;box-shadow:0 8px 25px rgba(99,102,241,0.3);
            ">
                <span style="font-size:1.8rem;line-height:1">\U0001f9ec</span>
            </div>
            <h2 style="margin:0;font-size:1rem;font-weight:700;
                       color:#F1F5F9 !important;letter-spacing:-0.01em">
                Cancer Mutation<br>Pathogenicity Predictor
            </h2>
            <p style="margin:0.3rem 0 0;font-size:0.7rem;color:#64748B !important;
                      text-transform:uppercase;letter-spacing:0.1em;font-weight:600">
                Multi-Omics Deep Learning
            </p>
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
            """<div style="display:flex;align-items:center;gap:8px;padding:0.4rem 0.8rem;
               background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);
               border-radius:10px">
               <div style="width:8px;height:8px;background:#10B981;border-radius:50%;
                    box-shadow:0 0 8px #10B981"></div>
               <span style="color:#34D399 !important;font-size:0.8rem;font-weight:600">
               API Connected</span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="display:flex;align-items:center;gap:8px;padding:0.4rem 0.8rem;
               background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);
               border-radius:10px">
               <div style="width:8px;height:8px;background:#EF4444;border-radius:50%"></div>
               <span style="color:#F87171 !important;font-size:0.8rem;font-weight:600">
               API Offline</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.caption(f"Endpoint: {client.api_url}")

page_module = PAGES[selected_page]
page_module.render(client)
