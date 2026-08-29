"""스미싱 JSON/JSONL 데이터 전용의 경량 학습 pipeline.

입력 레코드는 ``text``, ``send_hour``, ``has_url``, ``has_phone_num``,
``char_len``, ``category``, ``label``을 사용한다. 거래 이력이 없는 문자
데이터이므로 일반 송금 사기 Main Model과 분리해 학습한다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.anonymize import anonymize_text
from training.model_registry import CANDIDATE_DIR, write_candidate_metadata

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REQUIRED_COLUMNS = {"text", "send_hour", "has_url", "has_phone_num", "char_len", "category", "label"}
NUMERIC_COLUMNS = ["send_hour", "has_url", "has_phone_num", "char_len"]
OPTIONAL_NUMERIC_COLUMNS = ["has_urgent_word"]


def load_smishing_records(path: Path) -> pd.DataFrame:
    """JSON 배열 또는 한 줄에 한 JSON 객체인 JSONL을 읽고 schema를 검증한다."""
    if not path.exists():
        raise FileNotFoundError(f"Smishing dataset not found: {path}")
    text = path.read_text(encoding="utf-8")
    parsed: Any = json.loads(text) if path.suffix.lower() == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = parsed.get("data", [parsed])
    if not isinstance(parsed, list):
        raise ValueError("Dataset must be a JSON list or JSONL records")
    frame = pd.DataFrame(parsed)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Smishing dataset missing columns: {sorted(missing)}")
    frame = frame[list(REQUIRED_COLUMNS)].copy()
    for column in [*NUMERIC_COLUMNS, *[name for name in OPTIONAL_NUMERIC_COLUMNS if name in frame], "label"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame["text"].notna() & frame["category"].notna()
        & frame[NUMERIC_COLUMNS].notna().all(axis=1) & frame["send_hour"].between(0, 23)
        & frame["has_url"].isin([0, 1]) & frame["has_phone_num"].isin([0, 1])
        & frame["char_len"].ge(0) & frame["label"].isin([0, 1])
    )
    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column in frame:
            valid &= frame[column].isin([0, 1])
    data = frame.loc[valid].copy()
    if len(data) != len(frame):
        raise ValueError(f"Rejected {len(frame) - len(data)} invalid smishing records")
    if data["label"].nunique() < 2:
        raise ValueError("Smishing training requires both normal and fraud labels")
    # 전화번호 등 원문 식별자는 학습 전에 제거하고, 존재 여부 feature만 보존한다.
    data["text"] = data["text"].map(anonymize_text)
    return data.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


def build_smishing_model(random_seed: int = 42, numeric_columns: list[str] | None = None) -> Pipeline:
    """한국어 문자 문맥과 제공된 보조 feature를 함께 쓰는 경량 CPU 모델."""
    active_numeric_columns = numeric_columns or NUMERIC_COLUMNS
    preprocess = ColumnTransformer([
        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=8_000, sublinear_tf=True), "text"),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ("numeric", StandardScaler(with_mean=False), active_numeric_columns),
    ])
    return Pipeline([
        ("features", preprocess),
        # 작은 데이터에서 완전히 분리되는 feature가 있어도 과도한 계수가 생기지 않도록
        # 규제를 강화한 경량 solver를 사용한다.
        ("classifier", LogisticRegression(max_iter=1_000, C=0.5, solver="liblinear", class_weight="balanced", random_state=random_seed)),
    ])


def _metrics(labels: pd.Series, probabilities: Any, threshold: float = 0.5) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }


def run_smishing_pipeline(dataset_path: Path, random_seed: int = 42) -> dict[str, Any]:
    """JSON/JSONL을 학습해 candidate와 평가 report만 생성한다."""
    data = load_smishing_records(dataset_path)
    if len(data) < 20:
        raise ValueError("At least 20 validated records are required for train/test evaluation")
    train, test = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=random_seed)
    active_numeric_columns = [*NUMERIC_COLUMNS, *[name for name in OPTIONAL_NUMERIC_COLUMNS if name in data]]
    model = build_smishing_model(random_seed, active_numeric_columns).fit(train, train["label"])
    metrics = _metrics(test["label"], model.predict_proba(test)[:, 1])
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CANDIDATE_DIR / f"smishing_model_{version}.joblib"
    joblib.dump(model, model_path)
    report_path = REPORTS / f"smishing_model_{version}_metrics.json"
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "version": version, "status": "candidate", "model_type": "tfidf_char_ngram_logistic_regression",
        "dataset_size": len(data), "features": ["text", *active_numeric_columns, "category"],
        "metrics": metrics, "source_file": dataset_path.name, "synthetic_demo": False,
        "note": "Text-only smishing candidate. It is not a transaction fraud production model.",
    }
    write_candidate_metadata(f"smishing_{version}", metadata)
    return {"version": version, "model_path": str(model_path), "report_path": str(report_path), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight smishing classifier from JSON/JSONL")
    parser.add_argument("dataset", type=Path, help="JSON or JSONL file with smishing records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_smishing_pipeline(args.dataset, args.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()
