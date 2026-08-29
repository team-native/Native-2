"""API와 저장 모델을 연결한다. 원문을 저장하거나 출력하지 않는다."""
from __future__ import annotations
import json
from pathlib import Path
from app.features.behavior_features import behavior_features_from_input
from app.features.transaction_features import transaction_features_from_input
from app.inference.fraud_predictor import JoblibFraudPredictor
from app.inference.text_predictor import JoblibTextPredictor
from app.schemas.analysis import AnalyzeRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIR = PROJECT_ROOT / "models" / "production"

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
