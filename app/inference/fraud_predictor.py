"""주 금융사기 모델 예측기."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol
import joblib
import pandas as pd

class FraudPredictor(Protocol):
    def predict_probability(self, features: dict[str, float | int]) -> float: ...

class JoblibFraudPredictor:
    def __init__(self, model_path: Path, feature_names: list[str]) -> None:
        if not model_path.exists(): raise FileNotFoundError(f"Fraud model not found: {model_path}")
        self._model, self._feature_names = joblib.load(model_path), feature_names
    def predict_probability(self, features: dict[str, float | int]) -> float:
        row = pd.DataFrame([[features[name] for name in self._feature_names]], columns=self._feature_names)
        return float(self._model.predict_proba(row)[0][1])
