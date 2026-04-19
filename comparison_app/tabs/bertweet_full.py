"""BERTweet full fine-tuning : WaiYarAung."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATASETS = ["imdb", "sst2", "emotion"]


@st.cache_data
def _load_report(data_dir: Path, ds: str) -> dict:
    path = data_dir / f"WaiYarAung_{ds}_classification_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _per_class_df(report: dict) -> pd.DataFrame:
    """Flatten classification report into a per-class dataframe."""
    rows = []
    for key, val in report.items():
        if key in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(val, dict):
            rows.append({
                "class": key,
                "precision": val.get("precision"),
                "recall": val.get("recall"),
                "f1-score": val.get("f1-score"),
                "support": val.get("support"),
            })
    return pd.DataFrame(rows)


def render(data_dir: Path, assets_dir: Path) -> None:
    st.header("BERTweet (Full Fine-tune) · WaiYarAung")
    st.markdown(
        """
**Approach:** classical full fine-tuning of [`vinai/bertweet-base`](https://huggingface.co/vinai/bertweet-base) :
all 135M parameters are updated, unlike LoRA's ~0.7M.

**Config:** LR 2e-5, 4-6 epochs per dataset, early stopping on val loss.
        """
    )

    # ── Headline metrics ──
    st.subheader("Headline metrics per dataset")
    rows = []
    for ds in DATASETS:
        r = _load_report(data_dir, ds)
        if not r:
            continue
        rows.append({
            "Dataset": ds.upper(),
            "Test Accuracy": r.get("accuracy"),
            "Weighted F1": r.get("weighted avg", {}).get("f1-score"),
            "Macro F1": r.get("macro avg", {}).get("f1-score"),
            "Support (test size)": int(r.get("weighted avg", {}).get("support") or 0),
        })
    metrics_df = pd.DataFrame(rows)

    def _highlight_max(col: pd.Series) -> list[str]:
        if col.name in ["Test Accuracy", "Weighted F1", "Macro F1", "Support (test size)"]:
            best = col.max(skipna=True)
            return [
                "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;" if pd.notna(v) and v == best else ""
                for v in col
            ]
        return ["" for _ in col]

    st.dataframe(
        metrics_df.style.apply(_highlight_max, axis=0).format(
            {
                "Test Accuracy": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Support (test size)": "{:,.0f}",
            },
            na_rep=":",
        ),
        hide_index=True,
        use_container_width=True,
    )

    fig = px.bar(
        metrics_df,
        x="Dataset",
        y="Weighted F1",
        text="Weighted F1",
        labels={"Weighted F1": "F1 (weighted)"},
        height=360,
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0.85, 1.0], tickformat=".2%"), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-class breakdowns ──
    st.subheader("Per-class precision / recall / F1")
    st.caption("How the model performs on each class. Emotion is the most revealing : 6 unbalanced classes.")
    ds_pick = st.selectbox("Dataset", options=DATASETS, index=2, key="bertweet_ds")  # default emotion
    r = _load_report(data_dir, ds_pick)
    if r:
        pc = _per_class_df(r)
        melted = pc.melt(id_vars=["class", "support"], value_vars=["precision", "recall", "f1-score"],
                         var_name="Metric", value_name="Value")
        fig2 = px.bar(
            melted,
            x="class",
            y="Value",
            color="Metric",
            barmode="group",
            hover_data=["support"],
            title=f"{ds_pick.upper()} · per-class metrics (support shown on hover)",
            height=420,
        )
        fig2.update_layout(yaxis=dict(range=[0.5, 1.0], tickformat=".2%"), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Raw per-class table:**")

        def _highlight_per_class(col: pd.Series) -> list[str]:
            if col.name in ["precision", "recall", "f1-score", "support"]:
                best = col.max(skipna=True)
                return [
                    "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;" if pd.notna(v) and v == best else ""
                    for v in col
                ]
            return ["" for _ in col]

        st.dataframe(
            pc.style.apply(_highlight_per_class, axis=0).format(
                {
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:,.0f}",
                },
                na_rep=":",
            ),
            hide_index=True,
            use_container_width=True,
        )

    # ── Figures ──
    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for col, ds in zip(cols, DATASETS):
        p = assets_dir / "bertweet" / f"WaiYarAung_{ds}_confusion_matrix.png"
        if p.exists():
            with col:
                st.image(str(p), caption=f"Confusion matrix · {ds.upper()}", use_container_width=True)
        else:
            with col:
                st.info(f"No confusion-matrix figure found for {ds}")
