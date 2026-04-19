"""Overall comparison tab : all models across all datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


OWNER_LABELS = {
    "Shivaangii Jaiswal": "Shivaangii Jaiswal",
    "WaiYarAung": "WaiYarAung",
    "YU JUNCHENG": "YU JUNCHENG",
    "joannasj": "joannasj",
    "Kevin Wong": "Kevin Wong",
    "Wang Yuekai": "Wang Yuekai",
}

DATASET_ORDER = ["imdb", "sst2", "emotion"]


@st.cache_data
def _load(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "overall_metrics.csv")
    # Report F1 uses weighted by default; fall back to macro for multi-class (emotion)
    df["f1_report"] = df["test_f1_weighted"].fillna(df["test_f1_macro"])
    df["owner_label"] = df["owner"].map(OWNER_LABELS).fillna(df["owner"])
    return df


def _best_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Pick each owner's best model per dataset (by F1, else accuracy)."""
    df = df.copy()
    df["_rank_key"] = df["f1_report"].fillna(df["test_accuracy"])
    idx = df.groupby(["owner", "dataset"])["_rank_key"].idxmax()
    return df.loc[idx].drop(columns=["_rank_key"])


def render(data_dir: Path, assets_dir: Path) -> None:  # noqa: ARG001
    st.header("Overall Model Comparison")

    df = _load(data_dir)
    best = _best_per_dataset(df)

    imdb_best = best[best["dataset"] == "imdb"].sort_values("f1_report", ascending=False).head(1)
    sst2_best = best[best["dataset"] == "sst2"].sort_values("f1_report", ascending=False).head(1)
    emotion_best = best[best["dataset"] == "emotion"].sort_values("f1_report", ascending=False).head(1)

    highlights = ["**Key findings**"]
    if not imdb_best.empty:
        row = imdb_best.iloc[0]
        highlights.append(
            f"- **IMDb:** {row['model_display']} achieved the highest accuracy at **{row['test_accuracy']:.2%}**."
        )
    if not sst2_best.empty:
        row = sst2_best.iloc[0]
        highlights.append(
            f"- **SST-2:** {row['model_display']} achieved the highest accuracy at **{row['test_accuracy']:.2%}**."
        )
    if not emotion_best.empty:
        row = emotion_best.iloc[0]
        highlights.append(
            f"- **Emotion:** {row['model_display']} achieved the highest accuracy at **{row['test_accuracy']:.2%}**."
        )

    bertweet_full = df[df["model_display"] == "BERTweet (full fine-tune)"]
    bertweet_lora = df[df["model_display"] == "BERTweet LoRA r=8"]
    if not bertweet_full.empty and not bertweet_lora.empty:
        highlights.append(
            "- **BERTweet comparison:** full fine-tuning generally attains slightly higher accuracy, while LoRA provides a much more parameter-efficient alternative."
        )

    st.markdown("\n".join(highlights))

    # ───────────────── Per-dataset results (all rows) ─────────────────

    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds].sort_values("f1_report", ascending=False, na_position="last")
        if sub.empty:
            continue
        with st.expander(f"**{ds.upper()}** : {len(sub)} entries", expanded=(ds == "sst2")):
            cols = ["model_display", "method", "test_accuracy", "f1_report", "trainable_params"]
            display = sub[cols].rename(
                columns={
                    "model_display": "Model",
                    "method": "Approach",
                    "test_accuracy": "Accuracy",
                    "f1_report": "F1",
                    "trainable_params": "Trainable params",
                }
            )

            best_row_idx = set(display.index[display["F1"] == display["F1"].max(skipna=True)])

            def _highlight_leaderboard_cols(col: pd.Series) -> list[str]:
                if col.name in ["Accuracy", "F1"]:
                    best = col.max(skipna=True)
                    return [
                        "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;"
                        if pd.notna(v) and v == best else ""
                        for v in col
                    ]
                if col.name == "Trainable params":
                    best = col.min(skipna=True)
                    return [
                        "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;"
                        if pd.notna(v) and v == best else ""
                        for v in col
                    ]
                return ["" for _ in col]

            def _highlight_leaderboard_rows(row: pd.Series) -> list[str]:
                if row.name in best_row_idx:
                    return [
                        "background-color: rgba(46, 160, 67, 0.12); border-top: 1px solid rgba(46, 160, 67, 0.55); border-bottom: 1px solid rgba(46, 160, 67, 0.55);"
                        for _ in row
                    ]
                return ["" for _ in row]

            st.dataframe(
                display.style
                .apply(_highlight_leaderboard_rows, axis=1)
                .apply(_highlight_leaderboard_cols, axis=0)
                .format(
                    {
                        "Accuracy": "{:.4f}",
                        "F1": "{:.4f}",
                        "Trainable params": "{:,.0f}",
                    },
                    na_rep=":",
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.write("")

    # ───────────────── Side-by-side bar charts ─────────────────
    st.subheader("Model Accuracy")

    plot_df = df.copy()
    plot_df["metric"] = plot_df["test_accuracy"]
    plot_df["label"] = plot_df["model_display"]

    fig = px.bar(
        plot_df,
        x="dataset",
        y="metric",
        color="label",
        barmode="group",
        category_orders={"dataset": DATASET_ORDER},
        labels={"metric": "Test Accuracy", "dataset": "Dataset", "label": "Model"},
        height=520,
    )
    fig.update_layout(
        yaxis=dict(range=[0.75, 1.00], tickformat=".2%"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.divider()
    st.write("")

    # ───────────────── F1 comparison ─────────────────
    st.subheader("F1 comparison")
    f1_df = df.dropna(subset=["f1_report"])
    fig2 = px.bar(
        f1_df,
        x="dataset",
        y="f1_report",
        color="model_display",
        barmode="group",
        category_orders={"dataset": DATASET_ORDER},
        labels={"f1_report": "F1 (weighted / macro)", "dataset": "Dataset", "model_display": "Model"},
        hover_data=["model_display", "method", "notes"],
        height=500,
    )
    fig2.update_layout(
        yaxis=dict(range=[0.75, 1.00], tickformat=".2%"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.write("")
    st.divider()
    st.write("")

    # ───────────────── BERTweet comparison + analysis ─────────────────
    st.subheader("BERTweet with and without LoRA")

    target_models = ["BERTweet (full fine-tune)", "BERTweet LoRA r=8"]
    subset = df[df["model_display"].isin(target_models)].copy()

    if subset.empty:
        st.warning("No BERTweet comparison rows found in overall metrics.")
    else:
        fig3 = px.bar(
            subset,
            x="dataset",
            y="test_accuracy",
            color="model_display",
            barmode="group",
            text="test_accuracy",
            category_orders={"dataset": DATASET_ORDER, "model_display": target_models},
            labels={
                "dataset": "Dataset",
                "test_accuracy": "Test Accuracy",
                "model_display": "Model",
            },
            height=380,
        )
        fig3.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig3.update_layout(
            yaxis=dict(range=[0.75, 1.00], tickformat=".2%"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

        show = subset[
            ["dataset", "model_display", "test_accuracy", "f1_report", "trainable_params", "train_seconds", "peak_vram_mb"]
        ].rename(
            columns={
                "dataset": "Dataset",
                "model_display": "Model",
                "test_accuracy": "Accuracy",
                "f1_report": "F1",
                "trainable_params": "Trainable params",
                "train_seconds": "Train sec",
                "peak_vram_mb": "Peak VRAM (MB)",
            }
        )
        show["Dataset"] = show["Dataset"].str.upper()

        best_row_idx = set(show.index[show["F1"] == show["F1"].max(skipna=True)])

        def _highlight_bertweet_cols(col: pd.Series) -> list[str]:
            if col.name in ["Accuracy", "F1"]:
                best = col.max(skipna=True)
                return [
                    "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;"
                    if pd.notna(v) and v == best else ""
                    for v in col
                ]
            if col.name in ["Trainable params", "Train sec", "Peak VRAM (MB)"]:
                best = col.min(skipna=True)
                return [
                    "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;"
                    if pd.notna(v) and v == best else ""
                    for v in col
                ]
            return ["" for _ in col]

        def _highlight_bertweet_rows(row: pd.Series) -> list[str]:
            if row.name in best_row_idx:
                return [
                    "background-color: rgba(46, 160, 67, 0.12); border-top: 1px solid rgba(46, 160, 67, 0.55); border-bottom: 1px solid rgba(46, 160, 67, 0.55);"
                    for _ in row
                ]
            return ["" for _ in row]

        st.dataframe(
            show.style
            .apply(_highlight_bertweet_rows, axis=1)
            .apply(_highlight_bertweet_cols, axis=0)
            .format(
                {
                    "Accuracy": "{:.4f}",
                    "F1": "{:.4f}",
                    "Trainable params": "{:,.0f}",
                    "Train sec": "{:.1f}",
                    "Peak VRAM (MB)": "{:,.0f}",
                },
                na_rep=":",
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.write("")
    st.divider()
    st.write("")

    # ───────────────── Full raw table ─────────────────
    st.subheader("Full metrics table")

    model_summary = (
        df.groupby("model_display", as_index=False)
        .agg(
            mean_f1=("f1_report", "mean"),
            mean_acc=("test_accuracy", "mean"),
            mean_train_sec=("train_seconds", "mean"),
        )
        .sort_values(["mean_f1", "mean_acc", "mean_train_sec"], ascending=[False, False, True], na_position="last")
    )
    best_model_name = model_summary.iloc[0]["model_display"]
    st.caption(
        "Best overall model (by mean F1 across available datasets): "
        f"{best_model_name}"
    )

    display_all = df[
        [
            "owner_label",
            "model_display",
            "method",
            "dataset",
            "test_accuracy",
            "test_f1_weighted",
            "test_f1_macro",
            "trainable_params",
            "train_seconds",
            "peak_vram_mb",
            "notes",
        ]
    ].rename(
        columns={
            "owner_label": "Teammate",
            "model_display": "Model",
            "method": "Approach",
            "dataset": "Dataset",
            "test_accuracy": "Acc",
            "test_f1_weighted": "F1 (weighted)",
            "test_f1_macro": "F1 (macro)",
            "trainable_params": "Trainable",
            "train_seconds": "Train sec",
            "peak_vram_mb": "Peak VRAM (MB)",
            "notes": "Notes",
        }
    )

    best_row_idx = set(display_all.index[display_all["Model"] == best_model_name])
    higher_better_cols = ["Acc", "F1 (weighted)", "F1 (macro)"]
    lower_better_cols = ["Trainable", "Train sec", "Peak VRAM (MB)"]

    def _highlight_best(col: pd.Series) -> list[str]:
        if col.name in higher_better_cols:
            best = col.max(skipna=True)
            return [
                "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;"
                if pd.notna(v) and v == best else ""
                for v in col
            ]
        if col.name in lower_better_cols:
            best = col.min(skipna=True)
            return [
                "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;"
                if pd.notna(v) and v == best else ""
                for v in col
            ]
        return ["" for _ in col]

    def _highlight_best_model_row(row: pd.Series) -> list[str]:
        if row.name in best_row_idx:
            return [
                "background-color: rgba(46, 160, 67, 0.12); border-top: 1px solid rgba(46, 160, 67, 0.55); border-bottom: 1px solid rgba(46, 160, 67, 0.55);"
                for _ in row
            ]
        return ["" for _ in row]

    styler = (
        display_all.style
        .apply(_highlight_best_model_row, axis=1)
        .apply(_highlight_best, axis=0)
        .format(
            {
                "Acc": "{:.4f}",
                "F1 (weighted)": "{:.4f}",
                "F1 (macro)": "{:.4f}",
                "Trainable": "{:,.0f}",
                "Train sec": "{:.1f}",
                "Peak VRAM (MB)": "{:,.0f}",
            }
        )
    )
    st.dataframe(styler, hide_index=True, use_container_width=True)

