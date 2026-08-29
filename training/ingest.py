"""로컬/승인 데이터 ingest 확장 지점. 무단 크롤링은 수행하지 않는다."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class DatasetMetadata:
    """승인 데이터 출처·라이선스·검증 상태를 추적하기 위한 확장 계약."""
    dataset_source: str
    license: str
    collected_at: str
    verified: bool
def load_csv_files(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if not files: raise FileNotFoundError(f"No approved CSV dataset found in {directory}")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
