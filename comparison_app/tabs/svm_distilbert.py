"""TF-IDF + SVM and DistilBERT fine-tune : Wang Yuekai's models."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


METRICS = {
    "TF-IDF + SVM": {
        "imdb":    {"accuracy": 0.8915, "f1": 0.8913},
        "sst2":    {"accuracy": 0.8142, "f1": 0.8251},
        "emotion": {"accuracy": 0.8795, "f1": 0.8224},
    },
    "DistilBERT (fine-tuned)": {
        "imdb":    {"accuracy": 0.8810, "f1": 0.8823},
        "sst2":    {"accuracy": 0.8991, "f1": 0.9027},
        "emotion": {"accuracy": 0.9295, "f1": 0.8837},
    },
}


def render(data_dir: Path, assets_dir: Path) -> None:
    st.header("TF-IDF + SVM  ·  DistilBERT Fine-tune — Wang Yuekai")

    st.markdown(
        """
**Two models, one pipeline:**

- **TF-IDF + SVM** — fast, interpretable baseline. SVM coefficients show
  each word's weight toward the prediction. Treats words independently.
- **DistilBERT (fine-tuned)** — reads the full sentence at once.
  Consistently outperforms SVM, especially on short text (SST-2) and
  multi-class emotion where context matters most.

Also includes: cross-domain evaluation, robustness testing (typo / case /
negation), token-level explainability, and a YouTube Sentiment Radar.
        """
    )

    # ── Metrics table ──
    rows = []
    for model, datasets in METRICS.items():
        for ds, vals in datasets.items():
            rows.append({"Model": model, "Dataset": ds, **vals})
    df = pd.DataFrame(rows)

    st.subheader("In-domain metrics")
    st.dataframe(
        df.style.format({"accuracy": "{:.4f}", "f1": "{:.4f}"}),
        hide_index=True,
        use_container_width=True,
    )

    # ── Bar chart ──
    fig = px.bar(
        df, x="Dataset", y="f1", color="Model", barmode="group",
        title="F1 Score by Dataset",
        color_discrete_map={
            "TF-IDF + SVM": "#34d399",
            "DistilBERT (fine-tuned)": "#60a5fa",
        },
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # ── Key findings ──
    st.subheader("Key findings")
    st.markdown(
        """
- **IMDb (long text):** Both models perform similarly (~89%). Long reviews
  have enough keyword signal for bag-of-words to work well.
- **SST-2 (short text):** DistilBERT pulls ahead (90% vs 81%). Short
  sentences need contextual understanding.
- **Emotion (6-class):** Largest gap — DistilBERT 93% accuracy vs SVM 88%.
  Multi-class emotion requires semantic nuance.
- **Cross-domain:** DistilBERT drops 2–4 points across domains; SVM drops
  over 10. DistilBERT generalises better.
- **Robustness:** Case change has zero impact on both. Negation drops
  DistilBERT by 2.1% (it understands meaning); SVM barely reacts.
        """
    )
