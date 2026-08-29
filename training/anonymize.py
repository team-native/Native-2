"""학습 전 텍스트에서 명백한 식별자를 마스킹한다."""
import re
PHONE = re.compile(r"(?<!\d)01[016789][- ]?\d{3,4}[- ]?\d{4}(?!\d)")
RRN = re.compile(r"(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ACCOUNT = re.compile(r"(?<!\d)\d{2,4}[- ]\d{2,6}[- ]\d{2,6}(?!\d)")
LONG_NUMBER = re.compile(r"(?<!\d)\d{8,}(?!\d)")
def anonymize_text(text: object) -> str:
    value = str(text or "")
    for pattern, token in ((PHONE, "[PHONE]"), (RRN, "[RRN]"), (EMAIL, "[EMAIL]"), (ACCOUNT, "[ACCOUNT]"), (LONG_NUMBER, "[NUMBER]")): value = pattern.sub(token, value)
    return value
