"""XGBoost 기반 결합 금융사기 모델."""
from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier
from training.feature_engineering import MODEL_FEATURES

def train_fraud_model(frame: pd.DataFrame, config: dict[str, object]) -> XGBClassifier:
    if frame["label"].nunique() < 2: raise ValueError("Fraud model training requires both label classes")
    params, training = config["fraud_model"], config["training"]
    positives = int(frame["label"].sum()); negatives = len(frame) - positives
    model = XGBClassifier(n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]), learning_rate=float(params["learning_rate"]), subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]), scale_pos_weight=(negatives / positives if positives else 1.0), random_state=int(training["random_seed"]), eval_metric="logloss", n_jobs=1)
    return model.fit(frame[MODEL_FEATURES], frame["label"])

def save_fraud_model(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)
