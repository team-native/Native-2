"""API와 저장 모델을 연결한다. 원문을 저장하거나 출력하지 않는다."""
from __future__ import annotations
import json
import re
from pathlib import Path
from app.features.behavior_features import behavior_features_from_input
from app.features.transaction_features import transaction_features_from_input
from app.inference.fraud_predictor import JoblibFraudPredictor
from app.inference.text_predictor import JoblibTextPredictor
from app.inference.smishing_predictor import JoblibSmishingPredictor
from app.schemas.analysis import AnalyzeRequest, SmishingAnalyzeRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIR = PROJECT_ROOT / "models" / "production"
SMISHING_PRODUCTION_DIR = PROJECT_ROOT / "models" / "smishing" / "production"

class ModelNotReadyError(RuntimeError): pass

class InferenceService:
    def _load_metadata(self) -> dict[str, object]:
        path = PRODUCTION_DIR / "metadata.json"
        if not path.exists(): raise ModelNotReadyError("model_not_ready: production model has not been promoted")
        return json.loads(path.read_text(encoding="utf-8"))
    def is_ready(self) -> bool:
        return all((PRODUCTION_DIR / item).exists() for item in ("metadata.json", "text_model.joblib", "fraud_model.joblib"))
    def analyze(self, request: AnalyzeRequest) -> dict[str, object]:
        metadata = self._load_metadata()
        if not self.is_ready(): raise ModelNotReadyError("model_not_ready: production artifacts are incomplete")
        text_probability = JoblibTextPredictor(PRODUCTION_DIR / "text_model.joblib").predict_probability(request.text)
        features: dict[str, float | int] = {"text_fraud_probability": text_probability}
        features.update(transaction_features_from_input(**request.model_dump(include={"amount", "average_amount", "recipient_is_new", "recipient_transfer_count"})))
        features.update(behavior_features_from_input(**request.model_dump(include={"transfers_last_hour", "average_transfers_per_hour", "transfer_hour"})))
        probability = JoblibFraudPredictor(PRODUCTION_DIR / "fraud_model.joblib", list(metadata["features"])).predict_probability(features)
        return {"text_fraud_probability": text_probability, "fraud_probability": probability, "model_version": str(metadata["version"]), "risk_engine_applied": False, "llm_applied": False}

    def analyze_smishing(self, request: SmishingAnalyzeRequest) -> dict[str, object]:
        metadata_path = SMISHING_PRODUCTION_DIR / "metadata.json"
        model_path = SMISHING_PRODUCTION_DIR / "model.joblib"
        if not metadata_path.exists() or not model_path.exists():
            raise ModelNotReadyError("model_not_ready: smishing production model has not been promoted")
        text = request.text
        features = {
            "text": text,
            "send_hour": request.send_hour,
            "has_url": int(bool(re.search(r"https?://|www\\.", text, re.IGNORECASE))),
            "has_phone_num": int(bool(re.search(r"(?:01[016789]|0[2-6][1-5]?)[-.\\s]?\\d{3,4}[-.\\s]?\\d{4}", text))),
            "has_urgent_word": int(bool(re.search(r"긴급|즉시|바로|지금|당장", text))),
            "char_len": len(text),
            "category": request.category,
        }
        probability = JoblibSmishingPredictor(model_path).predict_probability(features)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"smishing_probability": probability, "model_version": str(metadata["version"]), "model_name": str(metadata.get("model_name", "공주"))}
