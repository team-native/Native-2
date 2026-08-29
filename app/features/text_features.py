"""텍스트 모델 입력을 위한 최소 전처리."""
import re

def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
