from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml_portfolio.common import RANDOM_STATE, binary_classification_metrics, ensure_artifact_dir, save_json, save_model


def make_credit_data(n_rows: int = 2600) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    income = rng.lognormal(mean=10.8, sigma=0.42, size=n_rows).clip(22000, 210000)
    debt_to_income = rng.beta(2.2, 5.0, n_rows).clip(0.02, 0.82)
    credit_score = rng.normal(690, 64, n_rows).clip(480, 835)
    loan_amount = rng.normal(18500, 8200, n_rows).clip(1500, 52000)
    delinquencies = rng.poisson(0.45, n_rows)
    employment = rng.choice(["full_time", "part_time", "self_employed", "unemployed"], n_rows, p=[0.62, 0.16, 0.16, 0.06])
    purpose = rng.choice(["debt_consolidation", "car", "home_improvement", "medical", "small_business"], n_rows)

    logit = (
        -2.6
        + 2.7 * debt_to_income
        - 0.0085 * (credit_score - 650)
        + 0.000018 * loan_amount
        - 0.000008 * income
        + 0.42 * delinquencies
        + 0.72 * (employment == "unemployed")
        + 0.42 * (employment == "self_employed")
        + 0.38 * (purpose == "small_business")
        + 0.24 * (purpose == "medical")
    )
    probability = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "income": income.round(2),
            "debt_to_income": debt_to_income.round(3),
            "credit_score": credit_score.round(0),
            "loan_amount": loan_amount.round(2),
            "delinquencies": delinquencies,
            "employment": employment,
            "purpose": purpose,
            "defaulted": defaulted,
        }
    )


def build_pipeline() -> Pipeline:
    numeric = ["income", "debt_to_income", "credit_score", "loan_amount", "delinquencies"]
    categorical = ["employment", "purpose"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, random_state=RANDOM_STATE)),
        ]
    )


def main() -> dict:
    df = make_credit_data()
    X = df.drop(columns=["defaulted"])
    y = df["defaulted"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.35).astype(int)
    metrics = binary_classification_metrics(y_test, y_pred, y_score)
    metrics["decision_threshold"] = 0.35

    artifacts = ensure_artifact_dir(__file__)
    save_model(model, artifacts / "credit_risk_model.joblib")
    save_json(metrics, artifacts / "metrics.json")
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
