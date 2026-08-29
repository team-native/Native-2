"""행동 데이터에서 모델용 수치 feature를 생성한다."""
from typing import Any
import numpy as np
import pandas as pd

DEFAULT_LATE_NIGHT_START = 0
DEFAULT_LATE_NIGHT_END = 5

def add_behavior_features(frame: pd.DataFrame, late_night_start: int = DEFAULT_LATE_NIGHT_START, late_night_end: int = DEFAULT_LATE_NIGHT_END) -> pd.DataFrame:
    result = frame.copy()
    recent = pd.to_numeric(result["transfers_last_hour"], errors="coerce")
    average = pd.to_numeric(result["average_transfers_per_hour"], errors="coerce")
    no_history = average.isna() | (average <= 0)
    result["has_behavior_history"] = (~no_history).astype(int)
    result["transfer_frequency_ratio"] = np.where(~no_history, recent / average, 0.0)
    hour = pd.to_numeric(result["transfer_hour"], errors="coerce").fillna(0).astype(int)
    result["transfer_hour"] = hour
    result["is_late_night"] = hour.between(late_night_start, late_night_end).astype(int)
    return result

def behavior_features_from_input(*, transfers_last_hour: float, average_transfers_per_hour: float, transfer_hour: int) -> dict[str, Any]:
    return add_behavior_features(pd.DataFrame([{"transfers_last_hour": transfers_last_hour, "average_transfers_per_hour": average_transfers_per_hour, "transfer_hour": transfer_hour}])).iloc[0].to_dict()
