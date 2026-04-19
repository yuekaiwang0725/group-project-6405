"""BiLSTM + Attention : joannasj."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATASETS = ["imdb", "sst2", "emotion"]
DS_PRETTY = {"imdb": "IMDB", "sst2": "SST2", "emotion": "Emotion"}


@st.cache_data
def _load(data_dir: Path) -> dict:
    path = data_dir / "bilstm_classification_reports.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _per_class_df(report: dict) -> pd.DataFrame:
    rows = []
    for cls, vals in report.get("classes", {}).items():
        rows.append({
            "class": cls,
            "precision": vals.get("precision"),
            "recall": vals.get("recall"),
            "f1-score": vals.get("f1-score"),
            "support": vals.get("support"),
        })
    return pd.DataFrame(rows)


def render(data_dir: Path, assets_dir: Path) -> None:
    st.header("BiLSTM + Attention · joannasj")
    st.markdown(
        """
**Approach:** bidirectional LSTM with attention, trained from scratch : the most
traditional deep sequence model in the project. Weights are saved at best
validation epoch via Keras checkpointing.
        """
    )

    data = _load(data_dir)
    if not data:
        st.warning("BiLSTM report file not found.")
        return

    st.caption(f"_Source: {data.get('source',':')}_")

    # Headline
    rows = []
    for ds in DATASETS:
        r = data.get(ds, {})
        if not r:
            continue
        rows.append({
            "Dataset": ds.upper(),
            "Test Accuracy": r.get("accuracy"),
            "Weighted F1": r.get("weighted avg", {}).get("f1-score"),
            "Macro F1": r.get("macro avg", {}).get("f1-score"),
            "Support (test)": int(r.get("weighted avg", {}).get("support") or 0),
        })
    metrics_df = pd.DataFrame(rows)

    st.subheader("Headline metrics per dataset")

    def _highlight_max(col: pd.Series) -> list[str]:
        if col.name in ["Test Accuracy", "Weighted F1", "Macro F1", "Support (test)"]:
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
                "Support (test)": "{:,.0f}",
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
        height=360,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0.75, 1.0], tickformat=".2%"), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Per-class
    st.subheader("Per-class precision / recall / F1")
    ds_pick = st.selectbox("Dataset", options=DATASETS, index=2, key="bilstm_ds")
    rep = data.get(ds_pick, {})
    pc = _per_class_df(rep)
    if not pc.empty:
        melted = pc.melt(id_vars=["class", "support"], value_vars=["precision", "recall", "f1-score"],
                         var_name="Metric", value_name="Value")
        fig2 = px.bar(
            melted,
            x="class",
            y="Value",
            color="Metric",
            barmode="group",
            hover_data=["support"],
            title=f"{ds_pick.upper()} · per-class metrics",
            height=420,
        )
        fig2.update_layout(yaxis=dict(range=[0.5, 1.0], tickformat=".2%"), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig2, use_container_width=True)

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

    # Figures
    st.subheader("Confusion matrices, loss curves, and LIME explanations")
    for ds in DATASETS:
        pretty = DS_PRETTY[ds]
        st.markdown(f"### {pretty}")
        cols = st.columns(3)
        for col, (suffix, label) in zip(
            cols,
            [("ConfusionMatrix", "Confusion matrix"), ("LossGraph", "Loss curve"), ("LIME", "LIME explanation")],
        ):
            p = assets_dir / "bilstm" / f"BiLSTM_AttentionMech_{pretty}_{suffix}.jpg"
            if p.exists():
                with col:
                    st.image(str(p), caption=f"{label} · {pretty}", use_container_width=True)
        # EvalMetrics image
        eval_p = assets_dir / "bilstm" / f"BiLSTM_AttentionMech_{pretty}_EvalMetrics.jpg"
        if eval_p.exists():
            st.image(str(eval_p), caption=f"Eval metrics (sklearn classification_report screenshot) · {pretty}", use_container_width=True)
