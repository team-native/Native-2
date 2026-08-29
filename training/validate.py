"""승인된 학습 데이터에만 적용할 schema 검증."""
from dataclasses import dataclass
import pandas as pd

REQUIRED_COLUMNS = ["case_id", "event_time", "text", "amount", "average_amount", "recipient_is_new", "recipient_transfer_count", "transfers_last_hour", "average_transfers_per_hour", "transfer_hour", "label", "label_source"]
NUMERIC_COLUMNS = ["amount", "average_amount", "recipient_transfer_count", "transfers_last_hour", "average_transfers_per_hour"]
@dataclass
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame

def validate_dataframe(frame: pd.DataFrame) -> ValidationResult:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing: raise ValueError(f"Dataset schema missing required columns: {missing}")
    data = frame.copy(); data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce", utc=True)
    for column in NUMERIC_COLUMNS + ["transfer_hour", "label"]: data[column] = pd.to_numeric(data[column], errors="coerce")
    if data["recipient_is_new"].dtype == object:
        data["recipient_is_new"] = data["recipient_is_new"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    valid_mask = (data["case_id"].notna() & data["event_time"].notna() & data["text"].notna() & data["label_source"].notna() & data["recipient_is_new"].notna() & data[NUMERIC_COLUMNS].notna().all(axis=1) & data[NUMERIC_COLUMNS].ge(0).all(axis=1) & data["transfer_hour"].between(0, 23) & data["label"].isin([0, 1]))
    return ValidationResult(data.loc[valid_mask].copy(), data.loc[~valid_mask].copy())
