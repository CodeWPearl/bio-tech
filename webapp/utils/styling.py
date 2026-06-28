"""CSS and styling helpers for the Streamlit dashboard."""

from __future__ import annotations

CLASS_COLORS: dict[str, str] = {
    "Pathogenic": "#EF4444",
    "Likely Pathogenic": "#F97316",
    "Benign": "#10B981",
    "Likely Benign": "#34D399",
}

CONFIDENCE_THRESHOLDS: list[tuple[float, str]] = [
    (0.8, "#10B981"),
    (0.6, "#F59E0B"),
    (0.0, "#EF4444"),
]


def get_class_color(predicted_class: str) -> str:
    """Return hex color for a pathogenicity class."""
    return CLASS_COLORS.get(predicted_class, "#6B7280")


def get_confidence_color(confidence: float) -> str:
    """Return green/yellow/red hex color based on confidence value."""
    for threshold, color in CONFIDENCE_THRESHOLDS:
        if confidence >= threshold:
            return color
    return "#EF4444"


def styled_metric_card(
    title: str,
    value: str,
    delta: str | None = None,
    icon: str = "",
    accent: str = "#6366F1",
) -> str:
    """Return HTML for a modern glassmorphism metric card."""
    delta_html = ""
    if delta is not None:
        delta_color = "#10B981" if delta.startswith("+") else "#94A3B8"
        delta_html = (
            f'<p style="color:{delta_color};font-size:0.75rem;margin:0;'
            f'font-weight:500">{delta}</p>'
        )

    icon_html = ""
    if icon:
        icon_html = f'<span style="font-size:1.6rem;display:block;margin-bottom:0.3rem">{icon}</span>'

    return f"""<div style="background:rgba(30,27,75,0.6);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px solid rgba(99,102,241,0.15);border-radius:16px;padding:1.3rem 1.2rem;text-align:center;box-shadow:0 4px 24px rgba(0,0,0,0.15);border-top:3px solid {accent}">{icon_html}<p style="color:#94A3B8;font-size:0.7rem;margin:0 0 0.4rem 0;text-transform:uppercase;letter-spacing:0.1em;font-weight:600">{title}</p><p style="color:#F1F5F9;font-size:1.6rem;font-weight:800;margin:0;line-height:1.2">{value}</p>{delta_html}</div>"""


def get_custom_css() -> str:
    """Return the custom CSS for the entire dashboard."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* Global */
        .stApp {
            background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 30%, #0F172A 70%, #1E293B 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #F1F5F9 !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stApp p, .stApp li, .stApp span, .stApp label {
            color: #CBD5E1 !important;
        }
        .stMarkdown a {
            color: #818CF8 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%);
            border-right: 1px solid rgba(99,102,241,0.15);
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] .stMarkdown label,
        section[data-testid="stSidebar"] .stMarkdown span {
            color: #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 2px;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #CBD5E1 !important;
            padding: 0.55rem 1rem !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(99,102,241,0.15) !important;
            color: white !important;
        }
        section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(99,102,241,0.15) !important;
        }

        /* Glass card */
        .glass-card {
            background: rgba(30,27,75,0.5);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(99,102,241,0.15);
            border-radius: 20px;
            padding: 1.8rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        .result-card {
            background: rgba(30,27,75,0.6);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(99,102,241,0.15);
            border-radius: 20px;
            padding: 1.8rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }

        /* Header */
        .dashboard-header {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #6366F1 100%);
            color: white;
            padding: 2.5rem 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(79,70,229,0.3);
            position: relative;
            overflow: hidden;
        }
        .dashboard-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            border-radius: 50%;
        }
        .dashboard-header::after {
            content: '';
            position: absolute;
            bottom: -30%;
            left: -10%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
            border-radius: 50%;
        }
        .dashboard-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
            position: relative;
            z-index: 1;
            letter-spacing: -0.02em;
        }
        .dashboard-header p {
            color: rgba(255,255,255,0.8) !important;
            margin: 0.6rem 0 0 0;
            font-size: 1.05rem;
            position: relative;
            z-index: 1;
            font-weight: 400;
        }

        /* Prediction badge */
        .prediction-badge {
            display: inline-block;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-weight: 800;
            font-size: 1.3rem;
            color: white;
            letter-spacing: 0.05em;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        /* Recommendation badge */
        .recommendation-badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.7rem;
        }
        .recommendation-high {
            background: rgba(16,185,129,0.15);
            color: #34D399;
            border: 1px solid rgba(16,185,129,0.3);
        }
        .recommendation-low {
            background: rgba(245,158,11,0.15);
            color: #FBBF24;
            border: 1px solid rgba(245,158,11,0.3);
        }

        /* Buttons */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            padding: 0.6rem 2rem;
            transition: all 0.2s ease;
            font-family: 'Inter', sans-serif;
        }
        .stButton > button[kind="primary"],
        .stButton > button:first-child {
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(99,102,241,0.3);
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button:first-child:hover {
            box-shadow: 0 6px 25px rgba(99,102,241,0.4) !important;
            transform: translateY(-1px);
        }

        /* Input fields */
        .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
            background: rgba(30,27,75,0.5) !important;
            border: 1px solid rgba(99,102,241,0.2) !important;
            border-radius: 10px !important;
            color: #E2E8F0 !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        }

        /* Checkbox */
        .stCheckbox label span {
            color: #CBD5E1 !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1rem;
            color: #E2E8F0 !important;
            background: rgba(30,27,75,0.3) !important;
            border-radius: 12px !important;
        }
        .streamlit-expanderContent {
            background: rgba(30,27,75,0.2) !important;
            border-radius: 0 0 12px 12px !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(30,27,75,0.3);
            border-radius: 12px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
            color: #94A3B8;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            color: white !important;
        }

        /* Dataframe */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: rgba(30,27,75,0.4);
            border: 1px solid rgba(99,102,241,0.1);
            border-radius: 12px;
            padding: 0.8rem;
        }
        [data-testid="stMetric"] label {
            color: #94A3B8 !important;
        }
        [data-testid="stMetricValue"] {
            color: #F1F5F9 !important;
        }

        /* Info/Warning/Error boxes */
        .stAlert {
            border-radius: 12px !important;
        }

        /* File uploader */
        .stFileUploader {
            background: rgba(30,27,75,0.3);
            border: 2px dashed rgba(99,102,241,0.3);
            border-radius: 16px;
            padding: 1rem;
        }

        /* Plotly charts dark theme */
        .js-plotly-plot .plotly .bg {
            fill: transparent !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(15,23,42,0.5);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(99,102,241,0.3);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(99,102,241,0.5);
        }

        /* Caption text */
        .stCaption, .stCaption p {
            color: #64748B !important;
        }

        /* Download buttons */
        .stDownloadButton > button {
            background: rgba(30,27,75,0.5) !important;
            border: 1px solid rgba(99,102,241,0.3) !important;
            color: #A5B4FC !important;
            border-radius: 12px !important;
        }
        .stDownloadButton > button:hover {
            background: rgba(99,102,241,0.15) !important;
            border-color: #6366F1 !important;
        }
    </style>
    """
