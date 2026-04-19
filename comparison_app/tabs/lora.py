"""LoRA tab: formal comparative study of parameter-efficient fine-tuning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATASET_ORDER = ["imdb", "sst2", "emotion"]

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
BINARY_CLASSES = ["negative", "positive"]


@st.cache_data
def _load_frontier(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "lora_efficiency_frontier.csv")
    df["model_short"] = df["model_name"].map(
        {
            "FacebookAI/roberta-large": "RoBERTa-large",
            "vinai/bertweet-base": "BERTweet",
            "distilbert-base-uncased": "DistilBERT",
        }
    )
    df["rank_label"] = "r=" + df["lora_r"].astype(int).astype(str)
    df["run_label"] = df["model_short"] + " · " + df["rank_label"]
    return df


@st.cache_data
def _load_bundle(data_dir: Path, run_name: str, dataset: str) -> dict:
    path = data_dir / f"{run_name}_{dataset}_bundle.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def _load_train_log(data_dir: Path, run_name: str, dataset: str) -> pd.DataFrame:
    path = data_dir / f"{run_name}_{dataset}_train_log.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _format_params(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(n):,}"


def _overview_block(df: pd.DataFrame) -> None:
    best_f1_row = df.loc[df["test_f1"].idxmax()]
    best_acc_row = df.loc[df["test_accuracy"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Models evaluated",
        f"{df['model_short'].nunique()}",
    )
    c1.caption(f"{', '.join(sorted(df['model_short'].unique()))}")
    c2.metric("Total runs", f"{len(df)}")
    c2.caption("Across IMDb, SST-2, and Emotion")
    c3.metric(
        "Best test accuracy",
        f"{best_acc_row['test_accuracy']:.4f}",
    )
    c3.caption(
        f"{best_acc_row['model_short']} {best_acc_row['rank_label']} · {str(best_acc_row['dataset']).upper()}"
    )
    c4.metric(
        "Best test F1",
        f"{best_f1_row['test_f1']:.4f}",
    )
    c4.caption(
        f"{best_f1_row['model_short']} {best_f1_row['rank_label']} · {str(best_f1_row['dataset']).upper()}"
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Mean test accuracy",
        f"{df['test_accuracy'].mean():.4f}",
    )
    c5.caption("Mean performance across all experimental runs")
    c6.metric(
        "Mean test precision",
        f"{df['test_precision'].mean():.4f}",
    )
    c6.caption("Average positive predictive value")
    c7.metric(
        "Mean test recall",
        f"{df['test_recall'].mean():.4f}",
    )
    c7.caption("Average sensitivity across datasets")
    c8.metric(
        "Avg trainable %",
        f"{df['trainable_ratio'].mean()*100:.2f}%",
    )
    c8.caption("Proportion of parameters updated during training")

def _headline_table(df: pd.DataFrame) -> None:
    st.subheader("Comprehensive Metrics Across All Experimental Runs")

    summary = (
        df.groupby(["model_short", "rank_label"], as_index=False)
        .agg(
            mean_f1=("test_f1", "mean"),
            mean_acc=("test_accuracy", "mean"),
            mean_train_sec=("train_seconds", "mean"),
        )
        .sort_values(["mean_f1", "mean_acc", "mean_train_sec"], ascending=[False, False, True])
    )
    best_model = summary.iloc[0]
    runner_up = summary.iloc[1] if len(summary) > 1 else None
    if runner_up is not None:
        delta_f1 = best_model["mean_f1"] - runner_up["mean_f1"]
        delta_acc = best_model["mean_acc"] - runner_up["mean_acc"]
        st.markdown(
            "**Best overall configuration:** "
            f"{best_model['model_short']} {best_model['rank_label']}. "
            "Using mean F1 across datasets as the primary criterion, this configuration achieved the "
            "strongest aggregate performance.\n\n"
            f"Relative to the next-ranked configuration ({runner_up['model_short']} {runner_up['rank_label']}), "
            f"it improves mean F1 by {delta_f1:.4f} and mean accuracy by {delta_acc:.4f}.\n\n"
            "One explanation is that RoBERTa-large has more capacity, and the higher LoRA rank gives "
            "it more flexibility to adapt to different tasks. At the same time, the chosen precision and "
            "regularization settings kept training stable."
        )
    else:
        st.markdown(
            "**Best overall configuration:** "
            f"{best_model['model_short']} {best_model['rank_label']}. "
            "This configuration is selected according to mean F1 across datasets, with mean accuracy and "
            "mean training time used as secondary criteria."
        )

    disp = df[
        [
            "model_short",
            "rank_label",
            "dataset",
            "test_accuracy",
            "test_f1",
            "test_precision",
            "test_recall",
            "trainable_params",
            "trainable_ratio",
            "train_seconds",
            "train_peak_vram_mb",
            "infer_peak_vram_mb",
            "bf16",
        ]
    ].copy()
    disp.columns = [
        "Model",
        "LoRA rank",
        "Dataset",
        "Acc",
        "F1",
        "Precision",
        "Recall",
        "Trainable",
        "Trainable %",
        "Train sec",
        "Train VRAM (MB)",
        "Infer VRAM (MB)",
        "bf16",
    ]
    disp["Precision Mode"] = np.where(disp["bf16"], "bf16", "fp16")
    disp = disp.drop(columns=["bf16"])

    # Sort by dataset then F1 desc
    order = {"imdb": 0, "sst2": 1, "emotion": 2}
    disp["_k"] = disp["Dataset"].map(order)
    disp = disp.sort_values(["_k", "F1"], ascending=[True, False]).drop(columns=["_k"])
    best_row_idx = set(
        disp.index[
            (disp["Model"] == best_model["model_short"]) &
            (disp["LoRA rank"] == best_model["rank_label"])
        ]
    )

    higher_better_cols = ["Acc", "F1", "Precision", "Recall"]
    lower_better_cols = ["Trainable", "Trainable %", "Train sec", "Train VRAM (MB)", "Infer VRAM (MB)"]

    def _highlight_best(col: pd.Series) -> list[str]:
        if col.name in higher_better_cols:
            best = col.max()
            return [
                "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;" if pd.notna(v) and v == best else ""
                for v in col
            ]
        if col.name in lower_better_cols:
            best = col.min()
            return [
                "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;" if pd.notna(v) and v == best else ""
                for v in col
            ]
        return ["" for _ in col]

    def _highlight_best_overall_row(row: pd.Series) -> list[str]:
        if row.name in best_row_idx:
            return [
                "background-color: rgba(46, 160, 67, 0.12); border-top: 1px solid rgba(46, 160, 67, 0.55); border-bottom: 1px solid rgba(46, 160, 67, 0.55);"
                for _ in row
            ]
        return ["" for _ in row]

    styler = (
        disp.style
        .apply(_highlight_best_overall_row, axis=1)
        .apply(_highlight_best, axis=0)
        .format(
            {
                "Acc": "{:.4f}",
                "F1": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "Trainable": "{:,.0f}",
                "Trainable %": "{:.2%}",
                "Train sec": "{:.1f}",
                "Train VRAM (MB)": "{:.0f}",
                "Infer VRAM (MB)": "{:.0f}",
            }
        )
    )
    st.dataframe(styler, hide_index=True, use_container_width=True)


def _rank_compare(df: pd.DataFrame) -> None:
    st.subheader("Comparative Analysis of LoRA Rank in RoBERTa-large ($r=8$ vs $r=16$)")

    rob = df[df["model_short"] == "RoBERTa-large"].copy()
    pivot_f1 = rob.pivot(index="dataset", columns="rank_label", values="test_f1")
    pivot_f1 = pivot_f1.reindex(DATASET_ORDER)
    pivot_f1["Δ F1"] = pivot_f1["r=16"] - pivot_f1["r=8"]
    pivot_time = rob.pivot(index="dataset", columns="rank_label", values="train_seconds").reindex(DATASET_ORDER)
    pivot_time["Δ sec"] = pivot_time["r=16"] - pivot_time["r=8"]
    pivot_vram = rob.pivot(index="dataset", columns="rank_label", values="train_peak_vram_mb").reindex(DATASET_ORDER)
    pivot_vram["Δ MB"] = pivot_vram["r=16"] - pivot_vram["r=8"]

    def _delta_style(v: float) -> str:
        if pd.isna(v):
            return ""
        if v > 0:
            return "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;"
        if v < 0:
            return "background-color: rgba(248, 81, 73, 0.22); color: #FFB3AD; font-weight: 600;"
        return "background-color: rgba(99, 110, 123, 0.20); color: #D0D7DE; font-weight: 600;"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Test-set F1 score**")
        st.dataframe(
            pivot_f1.round(4).style.map(_delta_style, subset=["Δ F1"]),
            use_container_width=True,
        )
    with c2:
        st.markdown("**Training wall-clock time (s)**")
        st.dataframe(
            pivot_time.round(1).style.map(_delta_style, subset=["Δ sec"]),
            use_container_width=True,
        )
    with c3:
        st.markdown("**Peak training VRAM (MB)**")
        st.dataframe(
            pivot_vram.round(0).style.map(_delta_style, subset=["Δ MB"]),
            use_container_width=True,
        )

    # Plotly grouped bar
    fig = go.Figure()
    for rank in ["r=8", "r=16"]:
        sub = rob[rob["rank_label"] == rank].sort_values("dataset")
        fig.add_trace(
            go.Bar(
                x=sub["dataset"],
                y=sub["test_f1"],
                name=rank,
                text=sub["test_f1"].round(4),
                textposition="outside",
            )
        )

    # Overlay analytical annotations directly on the chart.
    for ds in pivot_f1.index:
        if ds not in rob["dataset"].values:
            continue
        delta = pivot_f1.loc[ds, "Δ F1"]
        if pd.isna(delta):
            continue
        y_anchor = max(pivot_f1.loc[ds, "r=8"], pivot_f1.loc[ds, "r=16"]) + 0.003
        fig.add_annotation(
            x=ds,
            y=y_anchor,
            text=f"ΔF1 {delta:+.4f}",
            showarrow=False,
            font=dict(size=12, color="#E6EDF3"),
            bgcolor="rgba(27, 38, 59, 0.65)",
            bordercolor="rgba(230, 237, 243, 0.35)",
            borderwidth=1,
        )

    largest_ds = pivot_f1["Δ F1"].idxmax()
    y_largest = max(pivot_f1.loc[largest_ds, "r=8"], pivot_f1.loc[largest_ds, "r=16"]) + 0.011
    fig.add_annotation(
        x=largest_ds,
        y=y_largest,
        text=f"Largest improvement: {largest_ds.upper()}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.2,
        ax=0,
        ay=-35,
        font=dict(size=12),
    )

    fig.update_layout(
        title="RoBERTa-large LoRA: test-set F1 by rank and dataset",
        yaxis=dict(range=[0.80, 0.98], tickformat=".2%"),
        barmode="group",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**Interpretation:** Relative to $r=8$, rank $r=16$ improves performance on SST-2 (+0.3 F1) and Emotion (+1.0 F1),
while yielding no material gain on IMDb. The largest improvement occurs on Emotion, suggesting that the
6-class setting benefits from higher adapter capacity. For binary sentiment tasks, $r=8$ is already near
the empirical F1 ceiling, and additional rank primarily increases computational cost.
"""
    )


def _param_efficiency(df: pd.DataFrame) -> None:
    st.subheader("Parameter Efficiency: F1 per Million Trainable Parameters")
    st.caption(
        "Higher values indicate greater predictive return per trainable parameter. "
        "This view emphasizes the trade-off between adaptation capacity and parameter economy."
    )

    eff = df.copy()
    eff["f1_per_m_trainable"] = eff["test_f1"] / (eff["trainable_params"] / 1e6)
    fig = px.bar(
        eff.sort_values(["dataset", "f1_per_m_trainable"], ascending=[True, False]),
        x="run_label",
        y="f1_per_m_trainable",
        color="dataset",
        facet_col="dataset",
        category_orders={"dataset": DATASET_ORDER},
        labels={"f1_per_m_trainable": "F1 per million trainable params", "run_label": ""},
        height=440,
    )

    # Overlay key finding in each facet: highest parameter-efficiency configuration per dataset.
    for i, ds in enumerate(DATASET_ORDER, start=1):
        sub = eff[eff["dataset"] == ds]
        if sub.empty:
            continue
        best = sub.loc[sub["f1_per_m_trainable"].idxmax()]
        axis_suffix = "" if i == 1 else str(i)
        fig.add_annotation(
            x=best["run_label"],
            y=best["f1_per_m_trainable"] * 1.03,
            xref=f"x{axis_suffix}",
            yref=f"y{axis_suffix}",
            text=f"Highest efficiency: {best['run_label']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.1,
            ax=0,
            ay=-28,
            font=dict(size=11),
            bgcolor="rgba(27, 38, 59, 0.55)",
            bordercolor="rgba(230, 237, 243, 0.30)",
            borderwidth=1,
        )

    fig.update_xaxes(tickangle=-30)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=80))
    st.plotly_chart(fig, use_container_width=True)


def _training_curves(data_dir: Path, df: pd.DataFrame) -> None:
    st.subheader("Training Dynamics Across Epochs")
    st.caption(
        "The left panel reports training loss across optimization steps, and the right panel reports "
        "validation F1 by epoch. Early plateaus may indicate opportunities for shorter schedules, "
        "whereas continued upward trends at later epochs suggest potential under-training."
    )

    run_opts = sorted(df["run_name"].unique())
    selected_runs = st.multiselect(
        "Experimental runs to visualize (Cmd/Ctrl for multi-select)",
        options=run_opts,
        default=run_opts,
        key="lora_curve_runs",
    )
    selected_ds = st.multiselect(
        "Datasets",
        options=DATASET_ORDER,
        default=DATASET_ORDER,
        key="lora_curve_datasets",
    )

    if not selected_runs or not selected_ds:
        st.warning("Please select at least one run and one dataset.")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training loss (by step)", "Validation F1 (by epoch)"),
        horizontal_spacing=0.08,
    )
    for run_name in selected_runs:
        for ds in selected_ds:
            log = _load_train_log(data_dir, run_name, ds)
            if log.empty:
                continue
            trace_name = f"{run_name.replace('lora_','')} · {ds}"

            train = log.dropna(subset=["loss"])
            fig.add_trace(
                go.Scatter(
                    x=train["epoch"],
                    y=train["loss"],
                    mode="lines",
                    name=trace_name,
                    legendgroup=trace_name,
                    showlegend=True,
                ),
                row=1, col=1,
            )
            val = log.dropna(subset=["eval_f1"])
            if not val.empty:
                fig.add_trace(
                    go.Scatter(
                        x=val["epoch"],
                        y=val["eval_f1"],
                        mode="lines+markers",
                        name=trace_name,
                        legendgroup=trace_name,
                        showlegend=False,
                    ),
                    row=1, col=2,
                )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Eval F1", tickformat=".2%", row=1, col=2)
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _finetune_results(data_dir: Path, df: pd.DataFrame) -> None:
    st.subheader("Comparative Fine-tuning Outcomes")

    # A tidy table of Acc/Prec/Recall/F1 per (model, dataset)
    disp = df[["model_short", "rank_label", "dataset", "test_accuracy", "test_precision", "test_recall", "test_f1"]].copy()
    disp = disp.sort_values(["dataset", "model_short", "rank_label"]).reset_index(drop=True)

    # Grouped bar of all 4 metrics
    melted = disp.melt(
        id_vars=["model_short", "rank_label", "dataset"],
        value_vars=["test_accuracy", "test_precision", "test_recall", "test_f1"],
        var_name="Metric",
        value_name="Value",
    )
    melted["Metric"] = melted["Metric"].str.replace("test_", "").str.capitalize()
    melted["Run"] = melted["model_short"] + " " + melted["rank_label"]

    for ds in DATASET_ORDER:
        sub = melted[melted["dataset"] == ds]
        if sub.empty:
            continue
        fig = px.bar(
            sub,
            x="Run",
            y="Value",
            color="Metric",
            barmode="group",
            labels={"Value": "Score", "Run": ""},
            title=f"{ds.upper()}: accuracy, precision, recall, and F1",
            height=400,
        )

        # Overlay principal finding: highest F1 configuration for each dataset.
        f1_sub = sub[sub["Metric"] == "F1"]
        if not f1_sub.empty:
            best_f1 = f1_sub.loc[f1_sub["Value"].idxmax()]
            fig.add_annotation(
                x=best_f1["Run"],
                y=best_f1["Value"] + 0.032,
                text=f"Highest F1: {best_f1['Run']} ({best_f1['Value']:.4f})",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(27, 38, 59, 0.65)",
                bordercolor="rgba(230, 237, 243, 0.35)",
                borderwidth=1,
            )

        fig.update_layout(
            yaxis=dict(range=[0.75, 1.0], tickformat=".2%"),
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", y=-0.15, yanchor="top", x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)


def _fp16_case_study(data_dir: Path, assets_dir: Path, df: pd.DataFrame) -> None:
    st.subheader("Numerical Stability Case Study: bf16 to fp16 Transition")

    st.markdown(
        """
This diagnostic sequence summarizes the stability investigation for RoBERTa-large:

1. **Sweep v1** used LR $3\\times10^{-4}$ without warmup or gradient clipping.
    Four of twelve runs diverged with NaN values on SST-2 and Emotion.
2. **Sweep v2** introduced warmup and gradient clipping, and reduced LR to $2\\times10^{-4}$.
    IMDb stabilized (F1 = 0.931), but SST-2 and Emotion continued to diverge during warmup.
3. **Sweep v3** further reduced LR to $1\\times10^{-4}$ for smaller datasets.
    Divergence persisted at nearly identical training steps, indicating that LR alone was not causal.
4. **Sweep v4** replaced **bf16** with **fp16** (higher mantissa precision) and applied
    LR $5\\times10^{-5}$ with gradient clipping at 0.3. All runs converged without numerical failure.

These observations suggest that the training instability was mainly caused by limited numerical
precision, rather than by the learning rate alone. In simple terms, the model was producing values
that bf16 could not represent accurately enough during training. This became more noticeable when
the LoRA adapters generated large gradient updates, especially in sensitive components such as
LayerNorm. As a result, the training process became unstable and occasionally produced NaN values
(that is, invalid numerical results).

When the precision setting was changed to fp16 and dynamic loss scaling was enabled, the model had
more reliable numerical behavior during optimization. This reduced the chance of invalid values
appearing and allowed all runs to complete successfully.
        """
    )

    # Comparison table: unstable bf16 runs versus stabilized fp16 reruns
    case_df = pd.DataFrame({
        "Run": [
            "RoBERTa-large r=8 · SST-2",
            "RoBERTa-large r=16 · SST-2",
            "RoBERTa-large r=8 · Emotion",
            "RoBERTa-large r=16 · Emotion",
        ],
        "v1/v2/v3 (bf16)": ["NaN · F1=0.00", "NaN · F1=0.00", "NaN · F1=0.08", "NaN · F1=0.08"],
        "v4 (fp16) · Test F1": [
            f"{df[(df['run_name']=='lora_roberta_large_r8') & (df['dataset']=='sst2')]['test_f1'].iloc[0]:.4f}",
            f"{df[(df['run_name']=='lora_roberta_large_r16') & (df['dataset']=='sst2')]['test_f1'].iloc[0]:.4f}",
            f"{df[(df['run_name']=='lora_roberta_large_r8') & (df['dataset']=='emotion')]['test_f1'].iloc[0]:.4f}",
            f"{df[(df['run_name']=='lora_roberta_large_r16') & (df['dataset']=='emotion')]['test_f1'].iloc[0]:.4f}",
        ],
        "Precision update": ["bf16→fp16"] * 4,
        "LR": ["5e-5"] * 4,
        "Gradient clipping": ["0.3"] * 4,
    })

    case_df["_f1_numeric"] = case_df["v4 (fp16) · Test F1"].astype(float)

    def _highlight_case_cols(col: pd.Series) -> list[str]:
        if col.name == "_f1_numeric":
            best = col.max(skipna=True)
            return [
                "background-color: rgba(46, 160, 67, 0.25); color: #9BE9A8; font-weight: 600;" if pd.notna(v) and v == best else ""
                for v in col
            ]
        return ["" for _ in col]

    st.dataframe(
        case_df.style
        .apply(_highlight_case_cols, axis=0)
        .format({"_f1_numeric": "{:.4f}"})
        .hide(axis="columns", subset=["_f1_numeric"]),
        hide_index=True,
        use_container_width=True,
    )

    # Display representative stable curves from v4 fp16 reruns
    for run, ds in [("lora_roberta_large_r16", "sst2"), ("lora_roberta_large_r16", "emotion")]:
        log = _load_train_log(data_dir, run, ds)
        if log.empty:
            continue
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"{run.replace('lora_','')} · {ds} · training loss", f"{run.replace('lora_','')} · {ds} · validation F1"),
            horizontal_spacing=0.1,
        )
        train = log.dropna(subset=["loss"])
        val = log.dropna(subset=["eval_f1"])
        fig.add_trace(go.Scatter(x=train["epoch"], y=train["loss"], mode="lines", name="Train loss"), row=1, col=1)
        fig.add_trace(go.Scatter(x=val["epoch"], y=val["eval_f1"], mode="lines+markers", name="Val F1"), row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Val F1", tickformat=".2%", row=1, col=2)
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def _efficiency_frontier(assets_dir: Path, df: pd.DataFrame) -> None:
    st.subheader("Efficiency Frontier Visualizations")
    st.caption(
        "Precomputed frontier plots summarize trade-offs among predictive performance, training time, "
        "and memory consumption across all runs."
    )

    cols = st.columns(3)
    for col, ds in zip(cols, DATASET_ORDER):
        p = assets_dir / "lora" / f"efficiency_frontier_{ds}.png"
        if p.exists():
            with col:
                st.image(str(p), caption=f"Efficiency frontier · {ds.upper()}", use_container_width=True)

    st.markdown("**Dataset-wise decompositions of training time and VRAM utilization:**")
    for ds in DATASET_ORDER:
        c1, c2 = st.columns(2)
        for col, kind in zip([c1, c2], ["training_time", "vram"]):
            p = assets_dir / "lora" / f"{kind}_{ds}.png"
            if p.exists():
                with col:
                    st.image(str(p), caption=f"{kind.replace('_',' ').title()} · {ds.upper()}", use_container_width=True)

    st.markdown("**Analytical interpretation of the efficiency frontier:**")
    lines: list[str] = []
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds].copy()
        if sub.empty:
            continue
        best_f1 = sub.loc[sub["test_f1"].idxmax()]
        fastest = sub.loc[sub["train_seconds"].idxmin()]
        lowest_vram = sub.loc[sub["train_peak_vram_mb"].idxmin()]
        lines.append(
            f"- **{ds.upper()}**: The highest test F1 is obtained by {best_f1['model_short']} {best_f1['rank_label']} "
            f"({best_f1['test_f1']:.4f}). The shortest training time is achieved by {fastest['model_short']} "
            f"{fastest['rank_label']} ({fastest['train_seconds']/60:.2f} min), and the lowest peak training VRAM "
            f"is observed for {lowest_vram['model_short']} {lowest_vram['rank_label']} "
            f"({lowest_vram['train_peak_vram_mb']/1024:.2f} GiB)."
        )

    st.markdown(
        "\n".join(lines)
        + "\n\n"
        + "Across datasets, the frontier indicates a consistent trade-off: RoBERTa-large configurations "
        + "occupy the high-capacity region (higher memory and, in several settings, longer runtime), whereas "
        + "DistilBERT and BERTweet provide substantially lower resource requirements. Importantly, the "
        + "performance-optimal point is dataset-dependent: higher-capacity models are advantageous on SST-2, "
        + "while lighter backbones are competitive or superior on Emotion, suggesting that task complexity and "
        + "label structure materially affect the marginal utility of additional adaptation capacity."
    )


def _per_run_detail(data_dir: Path, assets_dir: Path, df: pd.DataFrame) -> None:
    st.subheader("Run-Level Analytical View")
    runs = df[["run_name", "dataset", "run_label"]].drop_duplicates()
    runs["picker"] = runs["run_label"] + " · " + runs["dataset"]
    pick = st.selectbox("Select an experimental run", options=runs["picker"].tolist(), key="lora_detail_pick")
    row = runs[runs["picker"] == pick].iloc[0]

    bundle = _load_bundle(data_dir, row["run_name"], row["dataset"])
    if not bundle:
        st.warning("No bundle metadata was found for the selected run.")
        return

    c1, c2, c3, c4 = st.columns(4)
    tm = bundle.get("test_metrics", {})
    c1.metric("Test Accuracy", f"{tm.get('test_accuracy', float('nan')):.4f}")
    c2.metric("Test F1", f"{tm.get('test_f1', float('nan')):.4f}")
    c3.metric("Test Precision", f"{tm.get('test_precision', float('nan')):.4f}")
    c4.metric("Test Recall", f"{tm.get('test_recall', float('nan')):.4f}")

    # Hyperparameters
    st.markdown("**Training configuration:**")
    hp = {
        "Model": bundle.get("model_name"),
        "LoRA rank": bundle.get("lora_r"),
        "LoRA α": bundle.get("lora_alpha"),
        "LoRA dropout": bundle.get("lora_dropout"),
        "Target modules": ", ".join(bundle.get("target_modules") or []),
        "Batch size": bundle.get("batch_size"),
        "Epochs (max)": bundle.get("epochs"),
        "Early stop patience": bundle.get("early_stopping_patience"),
        "Warmup ratio": bundle.get("warmup_ratio"),
        "Max grad norm": bundle.get("max_grad_norm"),
        "Learning rate": bundle.get("learning_rate"),
        "Precision": "bf16" if bundle.get("bf16") else "fp16" if bundle.get("fp16") else "fp32",
    }
    st.json(hp, expanded=False)

    # Profile
    st.markdown("**Computational profile:**")
    train_p = bundle.get("train_profile", {})
    param_p = bundle.get("param_stats", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Train sec", f"{train_p.get('seconds', 0):.1f}")
    c2.metric("Train peak VRAM", f"{train_p.get('peak_vram_mb', 0):.0f} MB")
    c3.metric("Trainable params", _format_params(param_p.get("trainable", 0)))

    # Pre-rendered training-curves image
    png = assets_dir / "lora" / f"{row['run_name']}_{row['dataset']}_training_curves.png"
    if png.exists():
        st.write("")
        st.image(str(png), use_container_width=True)


def render(data_dir: Path, assets_dir: Path) -> None:
    st.header("LoRA Fine-tuning Study · Shivaangii Jaiswal")
    st.markdown(
        "This section presents a systematic comparative study of LoRA-based parameter-efficient fine-tuning "
        "across three pretrained transformer backbones (RoBERTa-large, BERTweet, and DistilBERT) on IMDb, "
        "SST-2, and Emotion."
    )

    df = _load_frontier(data_dir)

    top_tabs = st.tabs(
        [
            "Study Overview",
            "Numerical Stability",
            "Run-Level Analysis",
        ]
    )

    with top_tabs[0]:
        _overview_block(df)
        st.divider()
        _headline_table(df)
        st.divider()
        _finetune_results(data_dir, df)
        st.divider()
        _rank_compare(df)

    with top_tabs[1]:
        _fp16_case_study(data_dir, assets_dir, df)

    with top_tabs[2]:
        _per_run_detail(data_dir, assets_dir, df)
