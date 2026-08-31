"""Candidate 기록, champion 비교, 사람 승인 기반 승격."""
import argparse
import json
import shutil
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR, PRODUCTION_DIR = PROJECT_ROOT / "models" / "candidates", PROJECT_ROOT / "models" / "production"
SMISHING_PRODUCTION_DIR = PROJECT_ROOT / "models" / "smishing" / "production"

def write_candidate_metadata(version: str, metadata: dict[str, Any]) -> Path:
    path = CANDIDATE_DIR / f"metadata_{version}.json"; path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"); return path

def compare_with_production(candidate_metrics: dict[str, Any]) -> dict[str, Any]:
    path = PRODUCTION_DIR / "metadata.json"
    if not path.exists(): return {"recommendation": "PROMOTION_CANDIDATE", "reason": "No production model exists"}
    current = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    changes = {key: float(candidate_metrics.get(key) or 0) - float(current.get(key) or 0) for key in ("recall", "precision", "f1", "pr_auc")}
    improves = all(value >= 0 for value in changes.values()) and any(value > 0 for value in changes.values())
    return {"recommendation": "PROMOTION_CANDIDATE" if improves else "REVIEW_REQUIRED", "changes": changes}

def promote(version: str) -> Path:
    metadata_path, text_path, fraud_path = (CANDIDATE_DIR / f"metadata_{version}.json", CANDIDATE_DIR / f"text_model_{version}.joblib", CANDIDATE_DIR / f"fraud_model_{version}.joblib")
    if not all(path.exists() for path in (metadata_path, text_path, fraud_path)): raise ValueError(f"Cannot promote {version}: candidate artifacts are incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")); PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text_path, PRODUCTION_DIR / "text_model.joblib"); shutil.copy2(fraud_path, PRODUCTION_DIR / "fraud_model.joblib")
    metadata["status"] = "production"; (PRODUCTION_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return PRODUCTION_DIR / "metadata.json"

def promote_smishing(version: str) -> Path:
    """Promote an approved smishing-only candidate without replacing the transfer model."""
    metadata_path = CANDIDATE_DIR / f"metadata_smishing_{version}.json"
    model_path = CANDIDATE_DIR / f"smishing_model_{version}.joblib"
    if not all(path.exists() for path in (metadata_path, model_path)):
        raise ValueError(f"Cannot promote smishing {version}: candidate artifacts are incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    SMISHING_PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, SMISHING_PRODUCTION_DIR / "model.joblib")
    metadata["status"] = "production"
    (SMISHING_PRODUCTION_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return SMISHING_PRODUCTION_DIR / "metadata.json"

def main() -> None:
    parser = argparse.ArgumentParser(); subs = parser.add_subparsers(dest="command", required=True)
    command = subs.add_parser("promote", help="승인된 candidate를 production으로 승격"); command.add_argument("version")
    smishing = subs.add_parser("promote-smishing", help="승인된 스미싱 candidate를 production으로 승격"); smishing.add_argument("version")
    args = parser.parse_args()
    if args.command == "promote": print(promote(args.version))
    if args.command == "promote-smishing": print(promote_smishing(args.version))
if __name__ == "__main__": main()
