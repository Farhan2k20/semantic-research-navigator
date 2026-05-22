from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml_portfolio.common import RANDOM_STATE, binary_classification_metrics, ensure_artifact_dir, save_json, save_model


POSITIVE = [
    "the setup was fast and the dashboard is easy to understand",
    "customer support solved my issue quickly",
    "the app saves our team hours every week",
    "pricing feels fair for the value",
    "the latest release is stable and polished",
    "reports are clear and useful for managers",
    "the workflow is simple and reliable",
    "excellent onboarding experience for new users",
    "the mobile experience works smoothly",
    "integrations connected without any trouble",
]

NEGATIVE = [
    "the product crashes when I upload files",
    "support took too long to respond",
    "the dashboard is confusing and slow",
    "pricing is too high for small teams",
    "the new release introduced several bugs",
    "reports are missing important filters",
    "the workflow requires too many manual steps",
    "onboarding was frustrating and unclear",
    "the mobile app freezes every morning",
    "integrations failed during setup",
]


def make_sentiment_data(repeats: int = 30) -> pd.DataFrame:
    rows = []
    modifiers = ["", " overall", " for our team", " after the update", " compared with last month"]
    for _ in range(repeats):
        for text in POSITIVE:
            for modifier in modifiers:
                rows.append({"review": f"{text}{modifier}", "positive": 1})
        for text in NEGATIVE:
            for modifier in modifiers:
                rows.append({"review": f"{text}{modifier}", "positive": 0})
    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def main() -> dict:
    df = make_sentiment_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["positive"], test_size=0.25, stratify=df["positive"], random_state=RANDOM_STATE
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = binary_classification_metrics(y_test, y_pred, y_score)

    artifacts = ensure_artifact_dir(__file__)
    save_model(model, artifacts / "sentiment_model.joblib")
    save_json(metrics, artifacts / "metrics.json")
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
