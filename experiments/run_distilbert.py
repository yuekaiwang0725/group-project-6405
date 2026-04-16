from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils.device import configure_device_runtime, resolve_device
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import classification_metrics, infer_average_mode
from src.utils.seed import set_seed
from src.visualization.train_curves import save_training_curves

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_run_name(model_name: str) -> str:
    if model_name == "distilbert-base-uncased":
        return "distilbert"
    return model_name.replace("/", "__").replace("-", "_")


def _display_name(model_name: str) -> str:
    if model_name == "distilbert-base-uncased":
        return "DistilBERT"
    if model_name == "bert-base-uncased":
        return "BERT-base-uncased"
    return model_name


def _load_split(dataset_name: str, split: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / f"{dataset_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}. Run `python -m src.main prepare-data` first."
        )
    return pd.read_csv(path)


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def run_distilbert(
    dataset_name: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    device: str = "auto",
    model_name: str = "distilbert-base-uncased",
    run_name: str | None = None,
) -> None:
    set_seed(42)
    selected_device = resolve_device(device)
    configure_device_runtime(selected_device)
    run_name = run_name or _default_run_name(model_name)
    model_label = _display_name(model_name)
    train_df = _load_split(dataset_name, "train")
    val_df = _load_split(dataset_name, "val")
    test_df = _load_split(dataset_name, "test")

    train_dataset = _to_hf_dataset(train_df)
    val_dataset = _to_hf_dataset(val_df)
    test_dataset = _to_hf_dataset(test_df)
    num_labels = int(train_df["label"].nunique())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    average_mode = infer_average_mode(train_df["label"].astype(int).tolist())

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return classification_metrics(
            y_true=labels.tolist(),
            y_pred=preds.tolist(),
            average=average_mode,
        )

    out_dir = ensure_dir(PROJECT_ROOT / "checkpoints" / run_name / dataset_name)
    results_tables = ensure_dir(PROJECT_ROOT / "results" / "tables")
    results_figures = ensure_dir(PROJECT_ROOT / "results" / "figures")
    logging_dir = ensure_dir(PROJECT_ROOT / "results" / "logs" / run_name)

    print(f"Running {model_label} on device: {selected_device}")
    print(f"Saving checkpoints/results under run name: {run_name}")

    args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        use_cpu=selected_device == "cpu",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))

    history_df = pd.DataFrame(trainer.state.log_history)
    history_df.to_csv(results_tables / f"{run_name}_{dataset_name}_train_log.csv", index=False)
    save_training_curves(
        history_df=history_df,
        output_path=results_figures / f"{run_name}_{dataset_name}_training_curves.png",
        title=f"{model_label} Training Curves ({dataset_name.upper()})",
    )
    write_json(test_metrics, results_tables / f"{run_name}_{dataset_name}_test_metrics.json")
    pd.DataFrame([test_metrics]).to_csv(
        results_tables / f"{run_name}_{dataset_name}_test_metrics.csv", index=False
    )
    history_df.tail(100).to_csv(logging_dir / f"{dataset_name}_recent_log_history.csv", index=False)

    print(f"{model_label} completed for {dataset_name}. Test metrics: {test_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for sentiment analysis")
    parser.add_argument("--dataset", default="imdb", choices=["imdb", "sst2", "emotion"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_distilbert(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        model_name=args.model_name,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
