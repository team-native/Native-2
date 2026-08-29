"""교체 가능한 텍스트 예측기 인터페이스와 TF-IDF 구현."""
from pathlib import Path
from typing import Protocol
import joblib
from app.features.text_features import normalize_text

class TextPredictor(Protocol):
    def predict_probability(self, text: str) -> float: ...

class JoblibTextPredictor:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists(): raise FileNotFoundError(f"Text model not found: {model_path}")
        self._model = joblib.load(model_path)
    def predict_probability(self, text: str) -> float:
        return float(self._model.predict_proba([normalize_text(text)])[0][1])
