"""학습과 추론이 공유하는 결합 feature 정의."""
import pandas as pd
from app.features.behavior_features import add_behavior_features
from app.features.transaction_features import add_transaction_features
MODEL_FEATURES = ["text_fraud_probability", "amount_ratio", "average_amount_missing", "recipient_is_new", "recipient_transfer_count", "has_transaction_history", "transfer_frequency_ratio", "has_behavior_history", "transfer_hour", "is_late_night"]
def engineer_structured_features(frame: pd.DataFrame) -> pd.DataFrame:
    return add_behavior_features(add_transaction_features(frame))
