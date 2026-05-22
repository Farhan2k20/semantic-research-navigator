from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml_portfolio.common import RANDOM_STATE, binary_classification_metrics, ensure_artifact_dir, save_json, save_model


def make_churn_data(n_rows: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    contract = rng.choice(["monthly", "annual", "two_year"], n_rows, p=[0.58, 0.29, 0.13])
    support_tickets = rng.poisson(1.4, n_rows)
    months_active = rng.integers(1, 61, n_rows)
    monthly_spend = rng.normal(72, 18, n_rows).clip(20, 160)
    product_usage = rng.beta(2.8, 2.0, n_rows)
    late_payments = rng.poisson(0.35, n_rows)
    discount = rng.choice(["none", "starter", "retention"], n_rows, p=[0.57, 0.28, 0.15])

    logit = (
        -1.4
        + 1.15 * (contract == "monthly")
        + 0.26 * support_tickets
        - 2.1 * product_usage
        - 0.025 * months_active
        + 0.35 * late_payments
        + 0.55 * (discount == "starter")
        - 0.4 * (discount == "retention")
        + 0.006 * (monthly_spend - 72)
    )
    probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "contract": contract,
            "support_tickets": support_tickets,
            "months_active": months_active,
            "monthly_spend": monthly_spend.round(2),
            "product_usage": product_usage.round(3),
            "late_payments": late_payments,
            "discount": discount,
            "churned": churned,
        }
    )


def build_pipeline() -> Pipeline:
    numeric = ["support_tickets", "months_active", "monthly_spend", "product_usage", "late_payments"]
    categorical = ["contract", "discount"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def main() -> dict:
    df = make_churn_data()
    X = df.drop(columns=["churned"])
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = binary_classification_metrics(y_test, y_pred, y_score)

    artifacts = ensure_artifact_dir(__file__)
    save_model(model, artifacts / "churn_model.joblib")
    save_json(metrics, artifacts / "metrics.json")
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
