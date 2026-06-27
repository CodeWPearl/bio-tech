"""CSS and styling helpers for the Streamlit dashboard."""

from __future__ import annotations

CLASS_COLORS: dict[str, str] = {
    "Pathogenic": "#DC3545",
    "Likely Pathogenic": "#FD7E14",
    "Benign": "#28A745",
    "Likely Benign": "#20C997",
}

CONFIDENCE_THRESHOLDS: list[tuple[float, str]] = [
    (0.8, "#28A745"),
    (0.6, "#FFC107"),
    (0.0, "#DC3545"),
]


def get_class_color(predicted_class: str) -> str:
    """Return hex color for a pathogenicity class."""
    return CLASS_COLORS.get(predicted_class, "#6C757D")


def get_confidence_color(confidence: float) -> str:
    """Return green/yellow/red hex color based on confidence value."""
    for threshold, color in CONFIDENCE_THRESHOLDS:
        if confidence >= threshold:
            return color
    return "#DC3545"


def styled_metric_card(title: str, value: str, delta: str | None = None) -> str:
    """Return HTML for a styled metric card."""
    delta_html = ""
    if delta is not None:
        delta_color = "#28A745" if delta.startswith("+") else "#DC3545"
        delta_html = f'<p style="color:{delta_color};font-size:0.85rem;margin:0">{delta}</p>'

    return f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1B6EC2;
        text-align: center;
    ">
        <p style="color:#6C757D;font-size:0.85rem;margin:0 0 0.3rem 0;
                  text-transform:uppercase;letter-spacing:0.05em">{title}</p>
        <p style="color:#212529;font-size:1.8rem;font-weight:700;margin:0">{value}</p>
        {delta_html}
    </div>
    """


def get_custom_css() -> str:
    """Return the custom CSS for the entire dashboard."""
    return """
    <style>
        /* Global overrides */
        .stApp {
            background-color: #F8F9FA;
        }
        .main .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1B3A5C 0%, #1B6EC2 100%);
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] .stMarkdown label {
            color: white !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: white !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.2);
        }

        /* Card styling */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }

        /* Prediction badge */
        .prediction-badge {
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.3rem;
            color: white;
            letter-spacing: 0.02em;
        }

        /* Recommendation badge */
        .recommendation-badge {
            display: inline-block;
            padding: 0.3rem 1rem;
            border-radius: 15px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .recommendation-high {
            background: #D4EDDA;
            color: #155724;
        }
        .recommendation-low {
            background: #FFF3CD;
            color: #856404;
        }

        /* Header styling */
        .dashboard-header {
            background: linear-gradient(135deg, #1B3A5C 0%, #1B6EC2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .dashboard-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2rem;
        }
        .dashboard-header p {
            color: rgba(255,255,255,0.85);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1rem;
        }

        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 2rem;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1B6EC2, #1B3A5C);
            color: white;
            border: none;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1.5rem;
        }
    </style>
    """
