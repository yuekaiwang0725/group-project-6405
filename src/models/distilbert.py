from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.device import configure_device_runtime, resolve_device


@dataclass
class DistilBertArtifacts:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification


def load_distilbert(model_name_or_path: str = "distilbert-base-uncased") -> DistilBertArtifacts:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
    return DistilBertArtifacts(tokenizer=tokenizer, model=model)


def load_finetuned_distilbert(model_dir: str | Path) -> DistilBertArtifacts:
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return DistilBertArtifacts(tokenizer=tokenizer, model=model)


def predict_sentiment(
    artifacts: DistilBertArtifacts, text: str, device: str | None = None
) -> tuple[int, float]:
    device = resolve_device(device)
    configure_device_runtime(device)

    model = artifacts.model.to(device)
    model.eval()
    inputs = artifacts.tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
    label = int(torch.argmax(probs).item())
    confidence = float(probs[label].item())
    return label, confidence
