"""승인된 raw 데이터에서 candidate만 생성하는 전체 재학습 pipeline."""
from __future__ import annotations
import json
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from training.evaluate import evaluate_predictions, save_evaluation
from training.feature_engineering import MODEL_FEATURES, engineer_structured_features
from training.ingest import load_csv_files
from training.model_registry import CANDIDATE_DIR, compare_with_production, write_candidate_metadata
from training.preprocess import preprocess_dataframe
from training.train_fraud_model import save_fraud_model, train_fraud_model
from training.train_text_model import save_text_model, train_text_model
from training.validate import validate_dataframe

ROOT = Path(__file__).resolve().parents[1]
REPORTS, VALIDATED, REJECTED, RAW = ROOT / "reports", ROOT / "data" / "validated", ROOT / "data" / "rejected", ROOT / "data" / "raw"

def load_config() -> dict[str, object]:
    with (ROOT / "configs" / "training.toml").open("rb") as handle: return tomllib.load(handle)

def _split(frame: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame["label"].nunique() < 2 or frame["case_id"].nunique() < 6: raise ValueError("Need at least 6 distinct case_id groups and both labels for leakage-safe splitting")
    settings = config["training"]; splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(settings["random_seed"]))
    train_val_index, test_index = next(splitter.split(frame, frame["label"], frame["case_id"]))
    train_val = frame.iloc[train_val_index].reset_index(drop=True); test = frame.iloc[test_index].reset_index(drop=True)
    splitter2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=int(settings["random_seed"]) + 1)
    train_index, validation_index = next(splitter2.split(train_val, train_val["label"], train_val["case_id"]))
    return train_val.iloc[train_index].reset_index(drop=True), train_val.iloc[validation_index].reset_index(drop=True), test

def run_pipeline(input_dir: Path | None = None) -> dict[str, object]:
    try:
        config = load_config()
        # raw에 승인 데이터가 아직 없으면, 저장소에 포함된 synthetic 검증용 데이터만 사용한다.
        source_dir = input_dir or (RAW if list(RAW.glob("*.csv")) else ROOT / "data" / "sample")
        frame = load_csv_files(source_dir); validation = validate_dataframe(frame)
        if validation.valid.empty: raise ValueError("No valid training rows after schema validation")
        VALIDATED.mkdir(parents=True, exist_ok=True); REJECTED.mkdir(parents=True, exist_ok=True)
        validation.valid.to_csv(VALIDATED / "latest_validated.csv", index=False); validation.rejected.to_csv(REJECTED / "latest_rejected.csv", index=False)
        data = preprocess_dataframe(validation.valid); train, validation_frame, test = _split(data, config)
        text_model = train_text_model(train, config)
        for split in (train, validation_frame, test): split["text_fraud_probability"] = text_model.predict_proba(split["text"])[:, 1]
        train, validation_frame, test = (engineer_structured_features(split) for split in (train, validation_frame, test))
        fraud_model = train_fraud_model(train, config)
        threshold = float(config["training"]["decision_threshold"])
        validation_metrics = evaluate_predictions(validation_frame["label"].to_numpy(), fraud_model.predict_proba(validation_frame[MODEL_FEATURES])[:, 1], threshold)
        test_metrics = evaluate_predictions(test["label"].to_numpy(), fraud_model.predict_proba(test[MODEL_FEATURES])[:, 1], threshold)
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        CANDIDATE_DIR.mkdir(parents=True, exist_ok=True); save_text_model(text_model, CANDIDATE_DIR / f"text_model_{version}.joblib"); save_fraud_model(fraud_model, CANDIDATE_DIR / f"fraud_model_{version}.joblib")
        # JSON report는 모든 환경에서 생성한다. 이미지 생성은 evaluate.save_evaluation의
        # 선택 기능으로 남겨 두어 headless/macOS font 환경이 학습을 막지 않게 한다.
        save_evaluation(test_metrics, REPORTS / f"model_{version}_metrics.json")
        metadata = {"version": version, "status": "candidate", "model_type": "xgboost", "text_model_type": "tfidf_logistic_regression", "dataset_size": len(data), "dataset_source": sorted(data["label_source"].astype(str).unique()), "features": MODEL_FEATURES, "feature_importance": dict(zip(MODEL_FEATURES, [float(value) for value in fraud_model.feature_importances_])), "metrics": test_metrics, "validation_metrics": validation_metrics, "decision_threshold": threshold, "synthetic_demo": bool((data["label_source"] == "synthetic_demo").all())}
        write_candidate_metadata(version, metadata); comparison = compare_with_production(test_metrics)
        result = {"version": version, "candidate_dir": str(CANDIDATE_DIR), "metrics": test_metrics, "comparison": comparison}; print(json.dumps(result, ensure_ascii=False, default=str)); return result
    except Exception as error:
        raise RuntimeError(f"Training pipeline failed: {error}") from error
if __name__ == "__main__": run_pipeline()
