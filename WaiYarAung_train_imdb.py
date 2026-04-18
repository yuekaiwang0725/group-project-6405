"""
BERTweet Sentiment Analysis on IMDB
- AdamW optimizer
- Lightweight hyperparameter tuning (learning rate search)
- Stratified K-Fold cross-validation
- Saves best model + evaluation artifacts for Streamlit
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "vinai/bertweet-base"
NUM_LABELS   = 2
MAX_LEN      = 128
BATCH_SIZE   = 16
NUM_EPOCHS   = 2
N_FOLDS      = 2
SUBSET_TRAIN = 600    # CPU-friendly; increase for better accuracy (e.g. 3000+)
SUBSET_TEST  = 200
RESULTS_DIR  = "results"

# Lightweight hyperparameter search space
LEARNING_RATES = [2e-5, 5e-5]
LABEL_NAMES    = ["Negative", "Positive"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "best_model"), exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Train / Eval helpers ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)

        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            truths.extend(lbls.cpu().numpy())
    return np.array(preds), np.array(truths)


def build_model_and_optimizer(lr, total_steps, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    return model, optimizer, scheduler


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data
    print("Loading IMDB ...")
    raw = load_dataset("imdb")
    # IMDB is sorted: first 12500 neg, last 12500 pos — take balanced samples
    half_tr = SUBSET_TRAIN // 2
    half_te = SUBSET_TEST  // 2
    train_texts  = raw["train"]["text"][:half_tr] + raw["train"]["text"][12500:12500+half_tr]
    train_labels = raw["train"]["label"][:half_tr] + raw["train"]["label"][12500:12500+half_tr]
    test_texts   = raw["test"]["text"][:half_te]  + raw["test"]["text"][12500:12500+half_te]
    test_labels  = raw["test"]["label"][:half_te] + raw["test"]["label"][12500:12500+half_te]

    # Load tokenizer (BERTweet requires use_fast=False)
    print("Loading BERTweet tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # ── Phase 1: Hyperparameter search via cross-validation ──────────────────
    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    cv_results = {}
    best_lr, best_cv_acc = None, 0.0

    for lr in LEARNING_RATES:
        print(f"\n{'='*55}")
        print(f"LR = {lr}")
        full_ds   = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        fold_accs = []

        for fold, (tr_idx, val_idx) in enumerate(
            skf.split(train_texts, train_labels)
        ):
            print(f"  Fold {fold + 1}/{N_FOLDS}")
            tr_loader  = DataLoader(Subset(full_ds, tr_idx), BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(Subset(full_ds, val_idx), BATCH_SIZE)

            total_steps = len(tr_loader) * NUM_EPOCHS
            model, optimizer, scheduler = build_model_and_optimizer(
                lr, total_steps, device
            )

            for epoch in range(NUM_EPOCHS):
                loss = train_epoch(model, tr_loader, optimizer, scheduler, device)
                print(f"    epoch {epoch+1}  loss={loss:.4f}")

            preds, truths = eval_epoch(model, val_loader, device)
            acc = (preds == truths).mean()
            fold_accs.append(float(acc))
            print(f"  -> fold acc: {acc:.4f}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_acc = float(np.mean(fold_accs))
        cv_results[lr] = {"fold_accs": fold_accs, "mean_acc": mean_acc}
        print(f"  CV mean acc: {mean_acc:.4f}")

        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lr     = lr

    print(f"\n{'='*55}")
    print(f"Best LR: {best_lr}  |  CV acc: {best_cv_acc:.4f}")

    # ── Phase 2: Train final model with best LR ───────────────────────────────
    print("\nTraining final model ...")
    train_ds = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    test_ds  = IMDBDataset(test_texts,  test_labels,  tokenizer, MAX_LEN)
    tr_loader  = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    tst_loader = DataLoader(test_ds,  BATCH_SIZE)

    total_steps = len(tr_loader) * NUM_EPOCHS
    model, optimizer, scheduler = build_model_and_optimizer(
        best_lr, total_steps, device
    )

    best_test_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        loss = train_epoch(model, tr_loader, optimizer, scheduler, device)
        preds, truths = eval_epoch(model, tst_loader, device)
        acc = (preds == truths).mean()
        print(f"Epoch {epoch+1}: loss={loss:.4f}  test_acc={acc:.4f}")

        if acc > best_test_acc:
            best_test_acc = float(acc)
            model.save_pretrained(os.path.join(RESULTS_DIR, "best_model"))
            tokenizer.save_pretrained(os.path.join(RESULTS_DIR, "best_model"))
            print("  -> Saved best model")

    # ── Phase 3: Evaluation artifacts ────────────────────────────────────────
    print("\nGenerating evaluation artifacts ...")
    preds, truths = eval_epoch(model, tst_loader, device)

    report_dict = classification_report(
        truths, preds, labels=[0, 1], target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )
    report_str = classification_report(truths, preds, labels=[0, 1], target_names=LABEL_NAMES, zero_division=0)
    print(report_str)

    cm = confusion_matrix(truths, preds, labels=[0, 1])

    # Confusion matrix PNG
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix — BERTweet on IMDB")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {cm_path}")

    # JSON results
    eval_results = {
        "best_lr":              best_lr,
        "best_cv_acc":          best_cv_acc,
        "test_accuracy":        best_test_acc,
        "cv_results":           {str(k): v for k, v in cv_results.items()},
        "classification_report":     report_dict,
        "classification_report_str": report_str,
        "val_texts":  test_texts[:100],
        "val_preds":  preds[:100].tolist(),
        "val_labels": truths[:100].tolist(),
    }
    results_path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "confusion_matrix.npy"), cm)

    print(f"\nAll artifacts saved to: {RESULTS_DIR}/")
    print("Training complete!")


if __name__ == "__main__":
    main()
