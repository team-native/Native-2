from pathlib import Path
import pytest

from training.model_registry import promote
from training.pipeline import ROOT, run_pipeline


def test_full_pipeline_creates_candidate_and_report() -> None:
    result = run_pipeline(ROOT / "data" / "sample")
    version = result["version"]
    assert Path(result["candidate_dir"], f"text_model_{version}.joblib").exists()
    assert (ROOT / "reports" / f"model_{version}_metrics.json").exists()
    assert result["metrics"]["confusion_matrix"].keys() == {"true_negative", "false_positive", "false_negative", "true_positive"}


def test_registry_rejects_missing_candidate() -> None:
    with pytest.raises(ValueError, match="artifacts are incomplete"):
        promote("does_not_exist")
