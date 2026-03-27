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
        return 1.0 / (1.0 + np.exp(-scores))
