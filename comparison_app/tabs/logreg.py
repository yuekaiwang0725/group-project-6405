"""TF-IDF + Logistic Regression : classical baseline (Kevin Wong)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATASETS = ["imdb", "sst2"]  # emotion was not run


@st.cache_data
def _load_report(data_dir: Path, ds: str) -> dict:
    path = data_dir / f"logistic_regression_{ds}_test.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def render(data_dir: Path, assets_dir: Path) -> None:
    st.header("TF-IDF + Logistic Regression · Kevin Wong")
    st.markdown(
        """
**Approach:** a classical baseline : tokens are turned into **TF-IDF** vectors and fed into
a **logistic regression** classifier. No neural network, no pretraining, no GPU.

**Why include it?** Every modern NLP experiment benefits from a strong non-neural baseline.
It sets the floor: any deep model that doesn't beat TF-IDF + LR is not worth the compute.
        """
    )

    # Headline metrics
    rows = []
    for ds in DATASETS:
        r = _load_report(data_dir, ds)
        if not r:
            continue
        rows.append({
            "Dataset": ds.upper(),
            "Accuracy": r.get("accuracy"),
            "Precision": r.get("precision"),
            "Recall": r.get("recall"),
            "F1": r.get("f1"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No logistic regression reports found.")
        return

    st.subheader("Headline metrics")

    def _highlight_max(col: pd.Series) -> list[str]:
        if col.name in ["Accuracy", "Precision", "Recall", "F1"]:
            best = col.max(skipna=True)
            return [
                "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;" if pd.notna(v) and v == best else ""
                for v in col
            ]
        return ["" for _ in col]

    st.dataframe(
        df.style.apply(_highlight_max, axis=0).format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
            },
            na_rep=":",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.info("**Emotion** (6-class) was not evaluated for this baseline : only the two binary sentiment tasks.")

    # Bar chart : side-by-side metrics
    long = df.melt(id_vars=["Dataset"], value_vars=["Accuracy", "Precision", "Recall", "F1"],
                   var_name="Metric", value_name="Score")
    fig = px.bar(
        long,
        x="Dataset",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        height=380,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0.70, 1.0], tickformat=".2f"),
                      margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # SHAP explainability
    st.subheader("SHAP explanations : what words drive the classifier?")
    st.markdown(
        "SHAP values attribute each prediction back to individual TF-IDF features. "
        "Since the model is linear on top of sparse tokens, the attributions are exact : "
        "a strength classical models have over deep ones."
    )

    cols = st.columns(len(DATASETS))
    for col, ds in zip(cols, DATASETS):
        img_path = assets_dir / "logreg" / f"logistic_regression_{ds}_shap.png"
        with col:
            st.markdown(f"**{ds.upper()}**")
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Missing: {img_path.name}")

    st.subheader("What this baseline tells us")
    st.markdown(
        """
- On **IMDb**, TF-IDF + LR hits **F1 = 0.882** : a very strong floor. Any neural model
  that lands below this on IMDb isn't earning its compute.
- On **SST-2**, F1 = 0.828. Short sentences with subtle sentiment markers hurt bag-of-words
  methods : this is where pretrained transformers pull ahead most.
- The classical pipeline trains in **seconds on CPU**, so it's an excellent sanity check
  before spending hours fine-tuning a transformer.
        """
    )
