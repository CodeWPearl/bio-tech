"""CSS and styling helpers for the Streamlit dashboard."""

from __future__ import annotations

THEME: dict[str, str] = {
    "bg": "#FAFAFA",
    "card": "#FFFFFF",
    "card_border": "#E2E8F0",
    "card_shadow": "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
    "text_primary": "#1E293B",
    "text_secondary": "#475569",
    "text_muted": "#64748B",
    "accent": "#4F46E5",
    "accent_light": "#EEF2FF",
    "accent_border": "#C7D2FE",
    "border": "#E2E8F0",
    "border_light": "#F1F5F9",
    "input_bg": "#FFFFFF",
    "input_border": "#E2E8F0",
    "sidebar_bg": "#FFFFFF",
    "sidebar_border": "#E2E8F0",
}

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

PLOTLY_LIGHT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#1E293B",
    font_family="Inter, sans-serif",
)


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
    accent: str = "#4F46E5",
) -> str:
    """Return HTML for a clean, minimal metric card."""
    delta_html = ""
    if delta is not None:
        delta_color = "#10B981" if delta.startswith("+") else "#64748B"
        delta_html = (
            f'<p style="color:{delta_color};font-size:0.75rem;margin:0;'
            f'font-weight:500">{delta}</p>'
        )

    icon_html = ""
    if icon:
        icon_html = f'<span style="font-size:1.6rem;display:block;margin-bottom:0.3rem">{icon}</span>'

    return (
        f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
        f"border-radius:12px;padding:1.3rem 1.2rem;text-align:center;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.06);border-top:3px solid {accent}\">"
        f"{icon_html}"
        f'<p style="color:#64748B;font-size:0.7rem;margin:0 0 0.4rem 0;'
        f'text-transform:uppercase;letter-spacing:0.1em;font-weight:600">{title}</p>'
        f'<p style="color:#1E293B;font-size:1.6rem;font-weight:800;margin:0;'
        f'line-height:1.2">{value}</p>'
        f"{delta_html}</div>"
    )


def get_custom_css() -> str:
    """Return the custom CSS for the entire dashboard."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* Global */
        .stApp {
            background: #FAFAFA;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #1E293B !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stApp p, .stApp li, .stApp span, .stApp label {
            color: #475569 !important;
        }
        .stMarkdown a {
            color: #4F46E5 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #1E293B !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] .stMarkdown label,
        section[data-testid="stSidebar"] .stMarkdown span {
            color: #475569 !important;
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 2px;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #475569 !important;
            padding: 0.55rem 1rem !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: #EEF2FF !important;
            color: #4F46E5 !important;
        }
        section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {
            background: #4F46E5 !important;
            color: white !important;
            box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #E2E8F0 !important;
        }

        /* Clean card */
        .glass-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.8rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .result-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.8rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        /* Header */
        .dashboard-header {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #4F46E5;
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .dashboard-header h1 {
            color: #1E293B !important;
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .dashboard-header p {
            color: #64748B !important;
            margin: 0.6rem 0 0 0;
            font-size: 1rem;
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
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
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
            background: rgba(16,185,129,0.1);
            color: #059669;
            border: 1px solid rgba(16,185,129,0.25);
        }
        .recommendation-low {
            background: rgba(245,158,11,0.1);
            color: #D97706;
            border: 1px solid rgba(245,158,11,0.25);
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 2rem;
            transition: all 0.2s ease;
            font-family: 'Inter', sans-serif;
        }
        .stButton > button[kind="primary"],
        .stButton > button:first-child {
            background: #4F46E5 !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(79,70,229,0.25);
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button:first-child:hover {
            background: #4338CA !important;
            box-shadow: 0 4px 12px rgba(79,70,229,0.3) !important;
            transform: translateY(-1px);
        }

        /* Input fields */
        .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            color: #1E293B !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: #4F46E5 !important;
            box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
        }

        /* Checkbox */
        .stCheckbox label span {
            color: #475569 !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1rem;
            color: #1E293B !important;
            background: #F8FAFC !important;
            border-radius: 10px !important;
        }
        .streamlit-expanderContent {
            background: #FFFFFF !important;
            border-radius: 0 0 10px 10px !important;
            border: 1px solid #E2E8F0 !important;
            border-top: none !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: #F1F5F9;
            border-radius: 10px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            color: #64748B;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #4F46E5;
            color: white !important;
        }

        /* Dataframe */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.8rem;
        }
        [data-testid="stMetric"] label {
            color: #64748B !important;
        }
        [data-testid="stMetricValue"] {
            color: #1E293B !important;
        }

        /* Info/Warning/Error boxes */
        .stAlert {
            border-radius: 10px !important;
        }

        /* File uploader */
        .stFileUploader {
            background: #FFFFFF;
            border: 2px dashed #CBD5E1;
            border-radius: 12px;
            padding: 1rem;
        }

        /* Plotly charts */
        .js-plotly-plot .plotly .bg {
            fill: transparent !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #F1F5F9;
        }
        ::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94A3B8;
        }

        /* Caption text */
        .stCaption, .stCaption p {
            color: #64748B !important;
        }

        /* Download buttons */
        .stDownloadButton > button {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            color: #4F46E5 !important;
            border-radius: 10px !important;
        }
        .stDownloadButton > button:hover {
            background: #EEF2FF !important;
            border-color: #4F46E5 !important;
        }
    </style>
    """
