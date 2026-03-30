"""Premium CSS injection for the Streamlit GUI.

Call ``inject_custom_css()`` once at the top of the Streamlit app to apply
a modern dark-theme aesthetic with glassmorphism cards, smooth gradients,
and micro-animations.
"""

from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Root variables ──────────────────────────────────────── */
:root {
    --bg-primary: #0f0f1a;
    --bg-card: rgba(30, 30, 60, 0.55);
    --bg-card-hover: rgba(40, 40, 80, 0.7);
    --border-glass: rgba(255, 255, 255, 0.08);
    --accent-green: #34d399;
    --accent-red: #f87171;
    --accent-blue: #60a5fa;
    --accent-purple: #a78bfa;
    --accent-amber: #fbbf24;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.35);
    --radius: 16px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global ──────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13132b 0%, #1a1a3e 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
}

/* ── Tab bar styling ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    transition: var(--transition) !important;
    color: var(--text-secondary) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(167, 139, 250, 0.3) !important;
}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius) !important;
    padding: 18px 22px !important;
    box-shadow: var(--shadow-card) !important;
    transition: var(--transition) !important;
}

[data-testid="stMetric"]:hover {
    background: var(--bg-card-hover) !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45) !important;
}

[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    font-size: 1.6rem !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 14px rgba(96, 165, 250, 0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(96, 165, 250, 0.4) !important;
}

/* ── Data frames ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden;
    border: 1px solid var(--border-glass) !important;
}

/* ── Text areas & inputs ─────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    backdrop-filter: blur(8px) !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2) !important;
}

/* ── Select boxes ────────────────────────────────────────── */
[data-baseweb="select"] {
    border-radius: 10px !important;
}

/* ── Expanders ───────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-glass) !important;
}

/* ── Alerts ──────────────────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Dashboard stat cards (custom HTML) ──────────────────── */
.stat-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 20px 24px;
    text-align: center;
    box-shadow: var(--shadow-card);
    transition: var(--transition);
}

.stat-card:hover {
    background: var(--bg-card-hover);
    transform: translateY(-3px);
    box-shadow: 0 12px 44px rgba(0, 0, 0, 0.5);
}

.stat-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 4px 0;
}

.stat-value.positive {
    background: linear-gradient(135deg, #34d399, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-value.negative {
    background: linear-gradient(135deg, #f87171, #fca5a5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-value.consensus {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-amber));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Comment card ────────────────────────────────────────── */
.comment-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: var(--transition);
}

.comment-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(167, 139, 250, 0.25);
}

.comment-text {
    color: var(--text-primary);
    font-size: 0.92rem;
    line-height: 1.5;
    margin-bottom: 6px;
}

.comment-meta {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-positive {
    background: rgba(52, 211, 153, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.badge-negative {
    background: rgba(248, 113, 113, 0.15);
    color: var(--accent-red);
    border: 1px solid rgba(248, 113, 113, 0.3);
}

.badge-emotion {
    background: rgba(167, 139, 250, 0.15);
    color: var(--accent-purple);
    border: 1px solid rgba(167, 139, 250, 0.3);
}

.confidence-bar {
    height: 4px;
    border-radius: 2px;
    background: rgba(255, 255, 255, 0.06);
    margin-top: 6px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.confidence-fill.positive {
    background: linear-gradient(90deg, var(--accent-green), #6ee7b7);
}

.confidence-fill.negative {
    background: linear-gradient(90deg, var(--accent-red), #fca5a5);
}

/* ── Dashboard header ────────────────────────────────────── */
.dashboard-header {
    text-align: center;
    padding: 20px 0 10px 0;
}

.dashboard-header h2 {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue), var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}

.dashboard-header p {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

/* ── Plotly chart containers ─────────────────────────────── */
.js-plotly-plot {
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Scrollable container ────────────────────────────────── */
.comments-scroll {
    max-height: 480px;
    overflow-y: auto;
    padding-right: 8px;
}

.comments-scroll::-webkit-scrollbar {
    width: 6px;
}

.comments-scroll::-webkit-scrollbar-track {
    background: transparent;
}

.comments-scroll::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 3px;
}

/* ── Section dividers ────────────────────────────────────── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glass), transparent);
    margin: 20px 0;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject premium dark-theme CSS into the current Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def stat_card(label: str, value: str, css_class: str = "") -> str:
    """Return HTML for a glassmorphism stat card."""
    return f"""
    <div class="stat-card">
        <div class="stat-value {css_class}">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """


def comment_card_html(
    text: str,
    sentiment: str | None = None,
    emotion: str | None = None,
    confidence: float | None = None,
    author: str | None = None,
) -> str:
    """Return HTML for a styled comment card with badges and a confidence bar."""
    badges = ""
    if sentiment:
        badge_cls = "badge-positive" if sentiment == "positive" else "badge-negative"
        icon = "👍" if sentiment == "positive" else "👎"
        badges += f'<span class="badge {badge_cls}">{icon} {sentiment}</span>'

    emotion_icons = {
        "sadness": "😢", "joy": "😄", "love": "❤️",
        "anger": "😠", "fear": "😨", "surprise": "😲",
    }
    if emotion:
        icon = emotion_icons.get(emotion, "🏷️")
        badges += f'<span class="badge badge-emotion">{icon} {emotion}</span>'

    if author:
        badges += f'<span style="color: var(--text-secondary); font-size: 0.75rem;">by {author}</span>'

    conf_bar = ""
    if confidence is not None:
        pct = min(100, max(0, confidence * 100))
        bar_cls = "positive" if sentiment == "positive" else "negative"
        conf_bar = f"""
        <div class="confidence-bar">
            <div class="confidence-fill {bar_cls}" style="width: {pct:.0f}%"></div>
        </div>
        """

    return f"""
    <div class="comment-card">
        <div class="comment-text">{text}</div>
        <div class="comment-meta">{badges}</div>
        {conf_bar}
    </div>
    """
