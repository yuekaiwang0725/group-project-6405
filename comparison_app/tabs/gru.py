"""Bi-GRU + Attention : YU JUNCHENG."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATASETS = ["imdb", "sst2", "emotion"]


@st.cache_data
def _load_report(data_dir: Path, ds: str) -> dict:
    path = data_dir / f"{ds}_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def render(data_dir: Path, assets_dir: Path) -> None:  # noqa: ARG001
    st.header("Bi-GRU + Attention · YU JUNCHENG")
    st.markdown(
        """
**Approach:** a bidirectional GRU with an attention pooling layer, trained **from scratch**
(no pretrained weights). Vocabulary of 8,000 tokens, 20 epochs.

**Trade-off:** faster to train per epoch than transformers and runs on CPU, but needs far
more epochs and cannot match transformer accuracy on complex tasks.
        """
    )

    # Headline
    rows = []
    for ds in DATASETS:
        r = _load_report(data_dir, ds)
        if not r:
            continue
        rows.append({
            "Dataset": ds.upper(),
            "Test Accuracy": r.get("test_accuracy", 0) / 100.0 if r.get("test_accuracy", 0) > 1 else r.get("test_accuracy"),
            "Total samples": r.get("total_samples"),
            "Architecture": r.get("model_architecture"),
            "Vocab size": r.get("vocab_size"),
            "Status": r.get("status"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No GRU reports found.")
        return

    display = df.copy()

    def _highlight(col: pd.Series) -> list[str]:
        if col.name == "Test Accuracy":
            best = col.max(skipna=True)
            return [
                "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;" if pd.notna(v) and v == best else ""
                for v in col
            ]
        if col.name == "Total samples":
            best = col.max(skipna=True)
            return [
                "background-color: rgba(46, 160, 67, 0.16); color: #9BE9A8;" if pd.notna(v) and v == best else ""
                for v in col
            ]
        return ["" for _ in col]

    st.subheader("Headline metrics")
    st.dataframe(
        display.style.apply(_highlight, axis=0).format(
            {
                "Test Accuracy": "{:.4f}",
                "Total samples": "{:,.0f}",
                "Vocab size": "{:,.0f}",
            },
            na_rep=":",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "Only **accuracy** is reported for this model (no per-class precision/recall/F1). "
        "For F1 comparisons, see the Overall tab."
    )

    # Bar chart
    fig = px.bar(
        df,
        x="Dataset",
        y="Test Accuracy",
        text="Test Accuracy",
        height=360,
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0.70, 1.0], tickformat=".2%"), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("What this model is")
    st.markdown(
        """
- **Bidirectional GRU** encodes each token in both directions
- **Attention pooling** learns a weighted sum over time-step hidden states instead of
  just taking the last state : helps with longer sequences
- **From-scratch vocabulary** of 8,000 tokens : no pretrained embeddings
- Fast to prototype and cheap to train, but hits a ceiling on nuanced tasks like Emotion
        """
    )
