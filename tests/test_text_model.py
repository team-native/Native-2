import pandas as pd
from training.pipeline import load_config
from training.train_text_model import train_text_model


def test_text_model_trains_and_predicts_probability() -> None:
    data = pd.DataFrame({"text": ["평소 관리비 송금", "검찰 안전계좌 즉시 송금", "점심 비용", "OTP 보내세요"], "label": [0, 1, 0, 1]})
    model = train_text_model(data, load_config())
    probability = model.predict_proba(["검찰이 즉시 송금하라고 합니다"])[0, 1]
    assert 0 <= probability <= 1
