import pandas as pd
import pytest

from training.validate import validate_dataframe


def row() -> dict[str, object]:
    return {"case_id": "a", "event_time": "2026-01-01T00:00:00Z", "text": "hello", "amount": 10, "average_amount": 5, "recipient_is_new": False, "recipient_transfer_count": 1, "transfers_last_hour": 1, "average_transfers_per_hour": 1, "transfer_hour": 10, "label": 0, "label_source": "manual_review"}


def test_validation_separates_invalid_rows() -> None:
    invalid = row(); invalid["transfer_hour"] = 25
    result = validate_dataframe(pd.DataFrame([row(), invalid]))
    assert len(result.valid) == 1 and len(result.rejected) == 1


def test_missing_schema_is_error() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_dataframe(pd.DataFrame([{"case_id": "x"}]))
