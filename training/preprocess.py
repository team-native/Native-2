"""검증된 데이터의 중복 제거 및 익명화."""
import pandas as pd
from app.features.text_features import normalize_text
from training.anonymize import anonymize_text
def preprocess_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(); result["text"] = result["text"].map(anonymize_text).map(normalize_text)
    return result.drop_duplicates().reset_index(drop=True)
