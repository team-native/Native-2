"""거래 데이터에서 모델용 수치 feature를 생성한다."""
from typing import Any
import numpy as np
import pandas as pd

def add_transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    average = pd.to_numeric(result["average_amount"], errors="coerce")
    amount = pd.to_numeric(result["amount"], errors="coerce")
    missing = average.isna() | (average <= 0)
    result["average_amount_missing"] = missing.astype(int)
    result["amount_ratio"] = np.where(~missing, amount / average, 0.0)
    result["recipient_is_new"] = result["recipient_is_new"].astype(bool).astype(int)
    result["recipient_transfer_count"] = pd.to_numeric(result["recipient_transfer_count"], errors="coerce").fillna(0.0)
    result["has_transaction_history"] = (result["recipient_transfer_count"] > 0).astype(int)
    return result

def transaction_features_from_input(*, amount: float, average_amount: float, recipient_is_new: bool, recipient_transfer_count: int) -> dict[str, Any]:
    return add_transaction_features(pd.DataFrame([{"amount": amount, "average_amount": average_amount, "recipient_is_new": recipient_is_new, "recipient_transfer_count": recipient_transfer_count}])).iloc[0].to_dict()
