from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC


class BaselineSVM:
    def __init__(self, c: float = 1.0) -> None:
        self.model = LinearSVC(C=c, random_state=42)

    def fit(self, x_train, y_train) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_confidence(self, x) -> np.ndarray:
        scores = self.model.decision_function(x)
        if scores.ndim == 1:
            # Binary: sigmoid gives P(positive). For negative preds, use 1 - p.
            proba_pos = 1.0 / (1.0 + np.exp(-scores))
            preds = self.model.predict(x)
            return np.where(preds == 1, proba_pos, 1.0 - proba_pos)
        # Multiclass: softmax-like per-row confidence
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
