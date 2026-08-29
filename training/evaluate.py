"""Recall 중심 평가와 JSON/혼동행렬 저장."""
from __future__ import annotations
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"threshold": threshold, "accuracy": accuracy_score(y_true, predicted), "precision": precision_score(y_true, predicted, zero_division=0), "recall": recall_score(y_true, predicted, zero_division=0), "f1": f1_score(y_true, predicted, zero_division=0), "roc_auc": roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else None, "pr_auc": average_precision_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else None, "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}}

def save_evaluation(metrics: dict[str, object], path: Path, image_path: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if image_path:
        values = metrics["confusion_matrix"]
        matrix = np.array([[values["true_negative"], values["false_positive"]], [values["false_negative"], values["true_positive"]]])
        figure, axis = plt.subplots(figsize=(4, 3)); axis.imshow(matrix, cmap="Blues")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"], xlabel="Predicted", ylabel="Actual")
        for row in range(2):
            for col in range(2): axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
        figure.tight_layout(); figure.savefig(image_path, dpi=150); plt.close(figure)
