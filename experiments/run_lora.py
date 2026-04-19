"""Experiment: fine-tune a transformer with LoRA (PEFT) and record efficiency metrics.

Mirrors the interface of ``experiments/run_distilbert.py`` so the main dashboard
and cross_domain/robustness runners can pick up the resulting checkpoint without
code changes. Adds a LoRA adapter via the `peft` library and attaches a
TrainingProfiler that logs peak VRAM + wall-clock to
``results/tables/<run_name>_<dataset>_profile.json``.

Typical invocation (see report/option_b_runbook.md for the full sweep):

    python -m experiments.run_lora \
        --dataset emotion \
        --model-name FacebookAI/roberta-large \
        --lora-r 16 \
        --lora-alpha 32 \
        --epochs 5 \
        --early-stopping-patience 1 \
        --batch-size 32 \
        --bf16

Design notes
------------
- Uses HuggingFace Trainer to keep parity with the rest of the repo.
- bf16 is opt-in but strongly recommended on H100 (A100 also supports it).
- `target_modules` defaults are conservative; override via --target-modules for
  non-RoBERTa backbones (e.g., for distilbert-base-uncased use q_lin,v_lin).
- Saves both the PEFT adapter (tiny) AND a merged full-weight checkpoint so
  the existing batch_predict / inference code can load it transparently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils.device import configure_device_runtime, resolve_device
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import classification_metrics, infer_average_mode
from src.utils.profiler import TrainingProfiler, count_trainable_params
from src.utils.seed import set_seed
from src.visualization.train_curves import save_training_curves

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model-family -> default LoRA target_modules. Override with --target-modules.
DEFAULT_TARGET_MODULES = {
    "roberta": ["query", "value"],
    "bert": ["query", "value"],
    "deberta": ["query_proj", "value_proj"],
    "distilbert": ["q_lin", "v_lin"],
    "bertweet": ["query", "value"],  # RoBERTa-based
}


def _infer_target_modules(model_name: str) -> list[str]:
    name = model_name.lower()
    for family, modules in DEFAULT_TARGET_MODULES.items():
        if family in name:
            return modules
    return ["query", "value"]  # sane fallback


def _default_run_name(model_name: str, lora_r: int) -> str:
    short = model_name.replace("/", "__").replace("-", "_")
    return f"lora_{short}_r{lora_r}"


def _load_split(dataset_name: str, split: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / f"{dataset_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}. Run `python -m src.main prepare-data` first."
        )
    return pd.read_csv(path)


def _to_hf(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def run_lora(
    dataset_name: str,
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str] | None,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    device: str = "auto",
    bf16: bool = False,
    fp16: bool = False,
    run_name: str | None = None,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int = 1,
    warmup_ratio: float = 0.06,
    max_grad_norm: float = 1.0,
) -> None:
    set_seed(42)
    selected_device = resolve_device(device)
    configure_device_runtime(selected_device)
    run_name = run_name or _default_run_name(model_name, lora_r)
    target_modules = target_modules or _infer_target_modules(model_name)

    train_df = _load_split(dataset_name, "train")
    val_df = _load_split(dataset_name, "val")
    test_df = _load_split(dataset_name, "test")
    num_labels = int(train_df["label"].nunique())

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    if gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    param_stats = count_trainable_params(model)
    model.print_trainable_parameters()
    print(f"[param_stats] {param_stats}")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = _to_hf(train_df).map(tokenize, batched=True)
    val_ds = _to_hf(val_df).map(tokenize, batched=True)
    test_ds = _to_hf(test_df).map(tokenize, batched=True)

    average_mode = infer_average_mode(train_df["label"].astype(int).tolist())

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return classification_metrics(
            y_true=labels.tolist(), y_pred=preds.tolist(), average=average_mode
        )

    out_dir = ensure_dir(PROJECT_ROOT / "checkpoints" / run_name / dataset_name)
    tables = ensure_dir(PROJECT_ROOT / "results" / "tables")
    figures = ensure_dir(PROJECT_ROOT / "results" / "figures")
    logs_dir = ensure_dir(PROJECT_ROOT / "results" / "logs" / run_name)

    # ── Sanity echo so we can see in logs what actually got used ─────
    print(
        f"[config] run={run_name} dataset={dataset_name} lr={learning_rate} "
        f"epochs={epochs} early_stop_patience={early_stopping_patience} "
        f"warmup_ratio={warmup_ratio} max_grad_norm={max_grad_norm} "
        f"bf16={bf16} fp16={fp16} bs={batch_size} seq={max_length}"
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        logging_dir=str(logs_dir),
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",   # MUST include the 'eval_' prefix or EarlyStoppingCallback silently disables itself
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        # fp16/bf16 are meaningless (and can crash) on CPU; force them off if we
        # fell back to CPU regardless of what the user asked for.
        bf16=bf16 and selected_device != "cpu",
        fp16=fp16 and selected_device != "cpu",
        gradient_checkpointing=gradient_checkpointing,
        use_cpu=selected_device == "cpu",
        seed=42,
        # dataloader workers: keep conservative, RunPod images tend to be narrow
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── profiled training ───────────────────────────────────────
    prof_device = f"cuda:{0}" if selected_device == "cuda" else selected_device
    profiler = TrainingProfiler(device=prof_device, label=f"{run_name}_{dataset_name}_train")
    with profiler:
        trainer.train()
    train_profile = profiler.stats

    # inference profile (separate — useful for latency story)
    inf_profiler = TrainingProfiler(device=prof_device, label=f"{run_name}_{dataset_name}_infer")
    with inf_profiler:
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    infer_profile = inf_profiler.stats

    # save adapter + merged full-weight (merged = plug-in compatible with Trainer)
    trainer.save_model(str(out_dir / "best_adapter"))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))

    # history + curves
    history_df = pd.DataFrame(trainer.state.log_history)
    history_df.to_csv(tables / f"{run_name}_{dataset_name}_train_log.csv", index=False)
    save_training_curves(
        history_df=history_df,
        output_path=figures / f"{run_name}_{dataset_name}_training_curves.png",
        title=f"{model_name} LoRA(r={lora_r}) — {dataset_name.upper()}",
    )

    # combined metrics + profile bundle
    bundle = {
        "run_name": run_name,
        "model_name": model_name,
        "dataset": dataset_name,
        "method": "lora",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "batch_size": batch_size,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "warmup_ratio": warmup_ratio,
        "max_grad_norm": max_grad_norm,
        "max_length": max_length,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
        "learning_rate": learning_rate,
        "param_stats": param_stats,
        "train_profile": train_profile,
        "infer_profile": infer_profile,
        "test_metrics": test_metrics,
    }
    write_json(bundle, tables / f"{run_name}_{dataset_name}_bundle.json")
    pd.DataFrame([test_metrics]).to_csv(
        tables / f"{run_name}_{dataset_name}_test_metrics.csv", index=False
    )
    write_json(test_metrics, tables / f"{run_name}_{dataset_name}_test_metrics.json")
    write_json(train_profile, tables / f"{run_name}_{dataset_name}_train_profile.json")
    write_json(infer_profile, tables / f"{run_name}_{dataset_name}_infer_profile.json")

    print(f"[done] {run_name} / {dataset_name}")
    print(f"  test_metrics = {test_metrics}")
    print(f"  train_profile = {train_profile}")
    print(f"  infer_profile = {infer_profile}")
    print(f"  trainable params = {param_stats['trainable']:,} / {param_stats['total']:,}")


def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA fine-tune for sentiment/emotion classification")
    p.add_argument("--dataset", default="emotion", choices=["imdb", "sst2", "emotion"])
    p.add_argument("--model-name", default="FacebookAI/roberta-large")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--target-modules", type=_csv_list, default=None,
                   help="Comma-separated layer names, e.g. 'query,value'.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=1,
                   help="Stop if val f1 doesn't improve for N epochs. 0 disables.")
    p.add_argument("--warmup-ratio", type=float, default=0.06,
                   help="LR warmup ratio (standard 0.06 for RoBERTa-style fine-tuning).")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Gradient clipping norm. 1.0 is a safe default, esp. for bf16.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="LoRA likes a higher LR than full fine-tuning (3e-4 is a good default).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on H100/A100).")
    p.add_argument("--fp16", action="store_true", help="Use fp16 (older GPUs).")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Trade compute for VRAM. Useful on 4090/24GB.")
    p.add_argument("--run-name", default=None)
    a = p.parse_args()
    run_lora(
        dataset_name=a.dataset,
        model_name=a.model_name,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        target_modules=a.target_modules,
        epochs=a.epochs,
        batch_size=a.batch_size,
        max_length=a.max_length,
        learning_rate=a.learning_rate,
        device=a.device,
        bf16=a.bf16,
        fp16=a.fp16,
        run_name=a.run_name,
        gradient_checkpointing=a.gradient_checkpointing,
        early_stopping_patience=a.early_stopping_patience,
        warmup_ratio=a.warmup_ratio,
        max_grad_norm=a.max_grad_norm,
    )


if __name__ == "__main__":
    main()
