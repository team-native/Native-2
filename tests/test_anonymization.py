from training.anonymize import anonymize_text


def test_masks_identifiers() -> None:
    text = anonymize_text("010-1234-5678 a@b.com 900101-1234567 123-456-7890")
    assert "010-1234-5678" not in text and "[PHONE]" in text
    assert "[EMAIL]" in text and "[RRN]" in text and "[ACCOUNT]" in text
