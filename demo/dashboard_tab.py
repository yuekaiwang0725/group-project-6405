"""Sentiment Radar — the Social Media Dashboard tab.

This module implements the '📊 Sentiment Radar' tab that is added to the
main gui_demo.  It is kept in its own file for readability and to avoid
making gui_demo.py excessively long.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from demo.custom_styles import comment_card_html, stat_card
from src.data.youtube_scraper import fetch_youtube_comments
from src.models.batch_predict import batch_predict_emotion, batch_predict_sentiment
from src.visualization.wordcloud_gen import (
    generate_emotion_wordcloud,
    generate_sentiment_wordcloud,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "demo" / "assets" / "demo_comments.json"

EMOTION_ICONS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

# Plotly theme defaults ──────────────────────────────────────
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=30, r=30, t=40, b=30),
)
_POS_COLOR = "#34d399"
_NEG_COLOR = "#f87171"
_EMOTION_COLORS = {
    "sadness": "#60a5fa",
    "joy": "#fbbf24",
    "love": "#f472b6",
    "anger": "#f87171",
    "fear": "#a78bfa",
    "surprise": "#fb923c",
}


# ── Data loading helpers ─────────────────────────────────────


def _load_demo_data() -> dict[str, list[dict[str, Any]]]:
    if DEMO_PATH.exists():
        with DEMO_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _extract_texts(comments: list[dict[str, Any]]) -> list[str]:
    return [c.get("text", "") for c in comments if c.get("text", "").strip()]


# ── Chart builders ───────────────────────────────────────────


def _sentiment_donut(sent_df: pd.DataFrame) -> go.Figure:
    if "distilbert_label" in sent_df.columns and sent_df["distilbert_label"].notna().any():
        model_col = "distilbert_label"
    elif "svm_label" in sent_df.columns and sent_df["svm_label"].notna().any():
        model_col = "svm_label"
    else:
        return go.Figure()
    counts = sent_df[model_col].value_counts()

    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.55,
            marker=dict(
                colors=[
                    _POS_COLOR if l == "positive" else _NEG_COLOR
                    for l in counts.index
                ],
                line=dict(color="rgba(0,0,0,0.3)", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, family="Inter"),
            hoverinfo="label+value+percent",
        )
    )
    fig.update_layout(
        title=dict(text="Sentiment Distribution", font=dict(size=16)),
        showlegend=False,
        **_PLOTLY_LAYOUT,
    )
    return fig


def _emotion_radar(emotion_df: pd.DataFrame) -> go.Figure:
    col = "distilbert_emotion"
    if col not in emotion_df.columns or emotion_df[col].isna().all():
        col = "svm_emotion"
    if col not in emotion_df.columns or emotion_df[col].isna().all():
        return go.Figure()

    counts = emotion_df[col].value_counts()
    categories = list(EMOTION_ICONS.keys())
    values = [int(counts.get(c, 0)) for c in categories]
    # Close the radar
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=[f"{EMOTION_ICONS[c]} {c}" for c in categories] + [f"{EMOTION_ICONS[categories[0]]} {categories[0]}"],
            fill="toself",
            fillcolor="rgba(167, 139, 250, 0.15)",
            line=dict(color="#a78bfa", width=2),
            marker=dict(size=6, color="#a78bfa"),
            name="Emotion",
        )
    )
    fig.update_layout(
        title=dict(text="Emotion Radar", font=dict(size=16)),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(color="#94a3b8"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=11, color="#e2e8f0"),
            ),
        ),
        showlegend=False,
        **_PLOTLY_LAYOUT,
    )
    return fig


def _model_comparison_bar(sent_df: pd.DataFrame) -> go.Figure:
    has_svm = "svm_label" in sent_df.columns and sent_df["svm_label"].notna().any()
    has_db = "distilbert_label" in sent_df.columns and sent_df["distilbert_label"].notna().any()
    if not has_svm and not has_db:
        return go.Figure()

    svm_pos = (sent_df["svm_label"] == "positive").sum()
    svm_neg = (sent_df["svm_label"] == "negative").sum()
    db_pos = (sent_df["distilbert_label"] == "positive").sum()
    db_neg = (sent_df["distilbert_label"] == "negative").sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Positive", x=["SVM", "DistilBERT"], y=[svm_pos, db_pos], marker_color=_POS_COLOR))
    fig.add_trace(go.Bar(name="Negative", x=["SVM", "DistilBERT"], y=[svm_neg, db_neg], marker_color=_NEG_COLOR))
    fig.update_layout(
        barmode="group",
        title=dict(text="Model Comparison", font=dict(size=16)),
        xaxis=dict(title="", tickfont=dict(size=12)),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_PLOTLY_LAYOUT,
    )
    return fig


def _emotion_bar(emotion_df: pd.DataFrame) -> go.Figure:
    col = "distilbert_emotion"
    if col not in emotion_df.columns or emotion_df[col].isna().all():
        col = "svm_emotion"
    if col not in emotion_df.columns:
        return go.Figure()

    counts = emotion_df[col].value_counts()
    categories = [c for c in EMOTION_ICONS if c in counts.index]
    values = [int(counts[c]) for c in categories]
    colors = [_EMOTION_COLORS.get(c, "#94a3b8") for c in categories]
    labels = [f"{EMOTION_ICONS.get(c, '')} {c}" for c in categories]

    fig = go.Figure(
        go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="auto")
    )
    fig.update_layout(
        title=dict(text="Emotion Breakdown", font=dict(size=16)),
        xaxis=dict(title=""),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.06)"),
        **_PLOTLY_LAYOUT,
    )
    return fig


# ── Main render function ─────────────────────────────────────


def render_dashboard_tab() -> None:
    """Render the full Sentiment Radar dashboard inside a Streamlit tab."""

    st.markdown(
        """
        <div class="dashboard-header">
            <h2>📊 Sentiment Radar — Social Media Dashboard</h2>
            <p>Analyse real-world comments using your trained NLP models in a complete end-to-end pipeline</p>
        </div>
        <div class="section-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    # ── Data source selection ────────────────────────────────
    source = st.radio(
        "Choose data source",
        ["🎬 Demo Data", "🔗 YouTube URL", "📄 Upload CSV"],
        horizontal=True,
        key="dashboard_source",
    )

    comments: list[dict[str, Any]] = []

    if source == "🎬 Demo Data":
        demo_data = _load_demo_data()
        demo_names = {
            "movie_reviews": "🎬 Movie Reviews",
            "product_reviews": "🛒 Product Reviews",
            "tech_discussion": "💻 Tech Discussion",
        }
        selected = st.selectbox(
            "Select demo dataset",
            list(demo_names.keys()),
            format_func=lambda k: demo_names[k],
            key="demo_select",
        )
        comments = demo_data.get(selected, [])

    elif source == "🔗 YouTube URL":
        col_url, col_count = st.columns([3, 1])
        with col_url:
            url = st.text_input(
                "YouTube video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                key="yt_url",
            )
        with col_count:
            max_count = st.number_input("Max comments", min_value=10, max_value=500, value=80, key="yt_max")
        if st.button("🔍 Fetch Comments", key="fetch_yt"):
            with st.spinner("Fetching YouTube comments…"):
                comments = fetch_youtube_comments(url, max_comments=int(max_count))
            if not comments:
                st.error("No comments found. Check the URL or try a different video.")
            else:
                st.success(f"Fetched {len(comments)} comments!")
            st.session_state["yt_comments"] = comments
        comments = st.session_state.get("yt_comments", [])

    else:  # CSV upload
        uploaded = st.file_uploader("Upload CSV (must have a 'text' column)", type=["csv"], key="csv_upload")
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            if "text" not in df_up.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                comments = [{"text": str(t), "author": "", "time": "", "likes": 0} for t in df_up["text"].tolist()]
                st.success(f"Loaded {len(comments)} rows from CSV.")

    if not comments:
        st.info("👆 Select a data source above to start analysing.")
        return

    # ── Run batch inference ──────────────────────────────────
    texts = _extract_texts(comments)
    if not texts:
        st.warning("No valid text found in the data.")
        return

    with st.spinner("🔄 Running sentiment & emotion analysis…"):
        sent_results = batch_predict_sentiment(texts)
        emotion_results = batch_predict_emotion(texts)

    sent_df = pd.DataFrame(sent_results)
    emotion_df = pd.DataFrame(emotion_results)

    # ── Overview stat cards ──────────────────────────────────
    total = len(texts)
    # Use whichever model is available for headline stats
    label_col = "distilbert_label" if sent_df["distilbert_label"].notna().any() else "svm_label"
    pos_count = int((sent_df[label_col] == "positive").sum()) if sent_df[label_col].notna().any() else 0
    neg_count = int((sent_df[label_col] == "negative").sum()) if sent_df[label_col].notna().any() else 0
    consensus_count = int(sent_df["consensus"].sum()) if sent_df["consensus"].notna().any() else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(stat_card("Total Comments", str(total)), unsafe_allow_html=True)
    with c2:
        pct = f"{pos_count / total * 100:.1f}%" if total else "0%"
        st.markdown(stat_card("Positive Rate", pct, "positive"), unsafe_allow_html=True)
    with c3:
        pct = f"{neg_count / total * 100:.1f}%" if total else "0%"
        st.markdown(stat_card("Negative Rate", pct, "negative"), unsafe_allow_html=True)
    with c4:
        pct = f"{consensus_count / total * 100:.1f}%" if total else "—"
        st.markdown(stat_card("Model Consensus", pct, "consensus"), unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Charts 2×2 grid ──────────────────────────────────────
    row1_left, row1_right = st.columns(2)
    with row1_left:
        fig_donut = _sentiment_donut(sent_df)
        st.plotly_chart(fig_donut, use_container_width=True, key="donut_chart")
    with row1_right:
        fig_radar = _emotion_radar(emotion_df)
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")

    row2_left, row2_right = st.columns(2)
    with row2_left:
        # Word cloud
        st.markdown("**💬 Sentiment Word Cloud**")
        pos_texts = sent_df.loc[sent_df[label_col] == "positive", "text"].tolist()
        neg_texts = sent_df.loc[sent_df[label_col] == "negative", "text"].tolist()
        wc_img = generate_sentiment_wordcloud(pos_texts, neg_texts)
        st.image(wc_img, use_container_width=True)

    with row2_right:
        fig_bar = _model_comparison_bar(sent_df)
        st.plotly_chart(fig_bar, use_container_width=True, key="compare_chart")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Emotion breakdown bar chart ──────────────────────────
    st.markdown("**🎭 Emotion Analysis**")
    emo_left, emo_right = st.columns(2)
    with emo_left:
        fig_emo = _emotion_bar(emotion_df)
        st.plotly_chart(fig_emo, use_container_width=True, key="emotion_bar")
    with emo_right:
        # Emotion word cloud
        st.markdown("**🌈 Emotion Word Cloud**")
        emo_col = "distilbert_emotion" if "distilbert_emotion" in emotion_df.columns and emotion_df["distilbert_emotion"].notna().any() else "svm_emotion"
        if emo_col in emotion_df.columns:
            texts_by_emotion: dict[str, list[str]] = {}
            for _, row in emotion_df.iterrows():
                emo = row.get(emo_col)
                if emo and isinstance(emo, str):
                    texts_by_emotion.setdefault(emo, []).append(str(row["text"]))
            emo_wc = generate_emotion_wordcloud(texts_by_emotion)
            st.image(emo_wc, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Comment list with filters ────────────────────────────
    st.markdown("**📝 Comment Details**")

    filter_col, sort_col = st.columns(2)
    with filter_col:
        filter_option = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive only", "Negative only"],
            key="comment_filter",
        )
    with sort_col:
        sort_option = st.selectbox(
            "Sort by",
            ["Default order", "Confidence (high → low)", "Confidence (low → high)"],
            key="comment_sort",
        )

    # Merge sentiment + emotion results
    display_data = []
    for i in range(len(texts)):
        entry = {**sent_results[i]}
        if i < len(emotion_results):
            entry.update({
                k: v for k, v in emotion_results[i].items()
                if k != "text"
            })
        if i < len(comments):
            entry["author"] = comments[i].get("author", "")
        display_data.append(entry)

    # Apply filter
    if filter_option == "Positive only":
        display_data = [d for d in display_data if d.get(label_col) == "positive"]
    elif filter_option == "Negative only":
        display_data = [d for d in display_data if d.get(label_col) == "negative"]

    # Apply sort
    conf_col = label_col.replace("_label", "_confidence")
    if sort_option == "Confidence (high → low)":
        display_data.sort(key=lambda d: d.get(conf_col, 0) or 0, reverse=True)
    elif sort_option == "Confidence (low → high)":
        display_data.sort(key=lambda d: d.get(conf_col, 0) or 0)

    # Render comment cards
    comments_html = '<div class="comments-scroll">'
    for d in display_data:
        sentiment = d.get(label_col, "")
        emo_key = "distilbert_emotion" if d.get("distilbert_emotion") else "svm_emotion"
        emotion = d.get(emo_key, "")
        confidence = d.get(conf_col)
        comments_html += comment_card_html(
            text=d.get("text", ""),
            sentiment=sentiment,
            emotion=emotion,
            confidence=confidence,
            author=d.get("author"),
        )
    comments_html += "</div>"
    st.markdown(comments_html, unsafe_allow_html=True)

    st.caption(f"Showing {len(display_data)} of {total} comments")
