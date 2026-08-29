import pandas as pd
from training.feature_engineering import MODEL_FEATURES
from training.pipeline import load_config
from training.train_fraud_model import train_fraud_model


def test_fraud_model_trains_and_predicts_probability() -> None:
    rows = []
    for label in (0, 1) * 4:
        rows.append({name: (0.9 if label else 0.1) if name == "text_fraud_probability" else (1 if label else 0) for name in MODEL_FEATURES} | {"label": label})
    model = train_fraud_model(pd.DataFrame(rows), load_config())
    probability = model.predict_proba(pd.DataFrame([rows[0]])[MODEL_FEATURES])[0, 1]
    assert 0 <= probability <= 1
