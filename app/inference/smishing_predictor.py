"""Production smishing model adapter."""
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class JoblibSmishingPredictor:
    def __init__(self, model_path: Path) -> None:
        self.model = joblib.load(model_path)

    def predict_probability(self, features: dict[str, Any]) -> float:
        return float(self.model.predict_proba(pd.DataFrame([features]))[:, 1][0])
