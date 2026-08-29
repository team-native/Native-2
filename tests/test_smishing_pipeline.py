import json

from training.smishing_pipeline import load_smishing_records, run_smishing_pipeline


def records() -> list[dict[str, object]]:
    result = []
    for index in range(12):
        result.append({"text": f"[Web발신] 정기 카드 사용 안내 {index}", "send_hour": 10, "has_url": 0, "has_phone_num": 0, "char_len": 25, "category": "정상안내", "label": 0})
        result.append({"text": f"검찰 긴급 대출 신청 안내 http://fake{index}.example 010-1234-{index:04d}", "send_hour": 2, "has_url": 1, "has_phone_num": 1, "char_len": 60, "category": "기관사칭", "label": 1})
    return result


def test_load_masks_phone_and_validates_records(tmp_path) -> None:
    path = tmp_path / "messages.jsonl"
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records()), encoding="utf-8")
    loaded = load_smishing_records(path)
    assert len(loaded) == 24
    assert "[PHONE]" in loaded.iloc[-1]["text"]


def test_smishing_pipeline_creates_candidate(tmp_path) -> None:
    path = tmp_path / "messages.json"
    path.write_text(json.dumps(records(), ensure_ascii=False), encoding="utf-8")
    result = run_smishing_pipeline(path)
    assert result["model_path"].endswith(".joblib")
    assert 0 <= result["metrics"]["recall"] <= 1
