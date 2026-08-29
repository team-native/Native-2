"""TF-IDF + Logistic Regression 텍스트 baseline."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_text_model(frame: pd.DataFrame, config: dict[str, object]) -> Pipeline:
    if frame["label"].nunique() < 2: raise ValueError("Text model training requires both label classes")
    settings = config["text_model"]
    return Pipeline([("tfidf", TfidfVectorizer(max_features=int(settings["max_features"]), ngram_range=(1, int(settings["ngram_max"])))), ("classifier", LogisticRegression(max_iter=int(settings["max_iter"]), class_weight="balanced", random_state=int(config["training"]["random_seed"]))) ]).fit(frame["text"], frame["label"])

def save_text_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)
