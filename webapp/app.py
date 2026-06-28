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
    cure_options,
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

if "client" not in st.session_state:
    st.session_state["client"] = APIClient()
client: APIClient = st.session_state["client"]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "\U0001f3e0  Home"

PAGES = {
    "\U0001f3e0  Home": home,
    "\U0001f52c  Single Prediction": single_prediction,
    "\U0001f4ca  Batch Analysis": batch_analysis,
    "\U0001f3e5  Cure Options": cure_options,
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
                background:#4F46E5;
                border-radius:14px;display:flex;align-items:center;
                justify-content:center;box-shadow:0 2px 8px rgba(79,70,229,0.2);
            ">
                <span style="font-size:1.8rem;line-height:1">\U0001f9ec</span>
            </div>
            <h2 style="margin:0;font-size:1rem;font-weight:700;
                       color:#1E293B !important;letter-spacing:-0.01em">
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
        index=list(PAGES.keys()).index(st.session_state["current_page"]),
        label_visibility="collapsed",
    )
    st.session_state["current_page"] = selected_page

    st.markdown("---")

    health = client.health_check()
    if health and health.get("status") == "healthy":
        st.markdown(
            """<div style="display:flex;align-items:center;gap:8px;padding:0.4rem 0.8rem;
               background:#F0FDF4;border:1px solid #BBF7D0;
               border-radius:10px">
               <div style="width:8px;height:8px;background:#10B981;border-radius:50%"></div>
               <span style="color:#059669 !important;font-size:0.8rem;font-weight:600">
               API Connected</span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="display:flex;align-items:center;gap:8px;padding:0.4rem 0.8rem;
               background:#FEF2F2;border:1px solid #FECACA;
               border-radius:10px">
               <div style="width:8px;height:8px;background:#EF4444;border-radius:50%"></div>
               <span style="color:#DC2626 !important;font-size:0.8rem;font-weight:600">
               API Offline</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#64748B !important;font-size:0.7rem;margin:0.4rem 0 0">'
            "Start the API server:<br>"
            '<code style="color:#4F46E5;font-size:0.65rem">'
            "uvicorn api.main:app --port 8001</code></p>",
            unsafe_allow_html=True,
        )

    st.caption(f"Endpoint: {client.api_url}")

page_module = PAGES[selected_page]
page_module.render(client)
