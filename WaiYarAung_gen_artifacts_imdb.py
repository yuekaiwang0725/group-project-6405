"""
Re-generates evaluation artifacts from the saved best model.
Run this after train.py saves the model weights.
"""
import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

RESULTS_DIR = "results"
MODEL_DIR   = os.path.join(RESULTS_DIR, "best_model")
LABEL_NAMES = ["Negative", "Positive"]
MAX_LEN     = 128
BATCH_SIZE  = 16
SUBSET_TEST = 200

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

raw        = load_dataset("imdb")
# IMDB test is sorted: first 12500 negative, last 12500 positive — take balanced sample
half       = SUBSET_TEST // 2
test_texts  = raw["test"]["text"][:half] + raw["test"]["text"][12500:12500+half]
test_labels = raw["test"]["label"][:half] + raw["test"]["label"][12500:12500+half]

loader = DataLoader(IMDBDataset(test_texts, test_labels, tokenizer, MAX_LEN), BATCH_SIZE)

preds, truths = [], []
with torch.no_grad():
    for batch in tqdm(loader, desc="eval"):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        out  = model(input_ids=ids, attention_mask=mask)
        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        truths.extend(lbls.cpu().numpy())

preds, truths = np.array(preds), np.array(truths)
print(f"Pred distribution: {np.bincount(preds)}")
print(f"True distribution: {np.bincount(truths)}")

report_dict = classification_report(truths, preds, labels=[0,1], target_names=LABEL_NAMES, output_dict=True, zero_division=0)
report_str  = classification_report(truths, preds, labels=[0,1], target_names=LABEL_NAMES, zero_division=0)
print(report_str)

cm = confusion_matrix(truths, preds, labels=[0, 1])
print("Confusion matrix:\n", cm)

# Load existing results and patch in eval fields
results_path = os.path.join(RESULTS_DIR, "eval_results.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        existing = json.load(f)
else:
    existing = {}

existing.update({
    "classification_report":     report_dict,
    "classification_report_str": report_str,
    "test_accuracy":             float((preds == truths).mean()),
    "val_texts":  test_texts[:100],
    "val_preds":  preds[:100].tolist(),
    "val_labels": truths[:100].tolist(),
})
with open(results_path, "w") as f:
    json.dump(existing, f, indent=2)

np.save(os.path.join(RESULTS_DIR, "confusion_matrix.npy"), cm)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
ax.set_title("Confusion Matrix -- BERTweet on IMDB")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

print("Artifacts saved to results/")
