import pandas as pd

from app.features.behavior_features import add_behavior_features
from app.features.transaction_features import add_transaction_features


def test_amount_ratio_and_missing_average() -> None:
    result = add_transaction_features(pd.DataFrame([{"amount": 100, "average_amount": 20, "recipient_is_new": True, "recipient_transfer_count": 0}, {"amount": 100, "average_amount": 0, "recipient_is_new": False, "recipient_transfer_count": 2}]))
    assert result.loc[0, "amount_ratio"] == 5 and result.loc[0, "average_amount_missing"] == 0
    assert result.loc[1, "amount_ratio"] == 0 and result.loc[1, "average_amount_missing"] == 1


def test_frequency_ratio_without_history() -> None:
    result = add_behavior_features(pd.DataFrame([{"transfers_last_hour": 4, "average_transfers_per_hour": 0.4, "transfer_hour": 2}, {"transfers_last_hour": 4, "average_transfers_per_hour": 0, "transfer_hour": 14}]))
    assert result.loc[0, "transfer_frequency_ratio"] == 10 and result.loc[0, "is_late_night"] == 1
    assert result.loc[1, "transfer_frequency_ratio"] == 0 and result.loc[1, "has_behavior_history"] == 0
