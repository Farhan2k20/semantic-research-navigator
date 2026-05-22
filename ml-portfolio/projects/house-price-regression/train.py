from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml_portfolio.common import RANDOM_STATE, ensure_artifact_dir, regression_metrics, save_json, save_model


def make_housing_data(n_rows: int = 2200) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    bedrooms = rng.integers(1, 6, n_rows)
    bathrooms = (bedrooms * rng.uniform(0.55, 0.9, n_rows) + rng.normal(0, 0.3, n_rows)).clip(1, 5).round(1)
    square_feet = (bedrooms * rng.normal(540, 95, n_rows) + rng.normal(450, 160, n_rows)).clip(600, 5200)
    age = rng.integers(0, 95, n_rows)
    neighborhood = rng.choice(["urban_core", "suburban", "waterfront", "rural"], n_rows, p=[0.32, 0.43, 0.08, 0.17])
    school_rating = rng.integers(3, 11, n_rows)
    commute_minutes = rng.normal(31, 12, n_rows).clip(5, 85)

    neighborhood_premium = pd.Series(neighborhood).map(
        {"urban_core": 55000, "suburban": 25000, "waterfront": 145000, "rural": -15000}
    ).to_numpy()
    price = (
        82000
        + square_feet * 178
        + bedrooms * 11500
        + bathrooms * 18500
        - age * 1200
        + school_rating * 14500
        - commute_minutes * 850
        + neighborhood_premium
        + rng.normal(0, 38000, n_rows)
    ).clip(90000, 1200000)

    return pd.DataFrame(
        {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "square_feet": square_feet.round(0),
            "age": age,
            "neighborhood": neighborhood,
            "school_rating": school_rating,
            "commute_minutes": commute_minutes.round(1),
            "sale_price": price.round(2),
        }
    )


def build_pipeline() -> Pipeline:
    categorical = ["neighborhood"]
    numeric = ["bedrooms", "bathrooms", "square_feet", "age", "school_rating", "commute_minutes"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=220, min_samples_leaf=4, random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )


def main() -> dict:
    df = make_housing_data()
    X = df.drop(columns=["sale_price"])
    y = df["sale_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    artifacts = ensure_artifact_dir(__file__)
    save_model(model, artifacts / "price_model.joblib")
    save_json(metrics, artifacts / "metrics.json")
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
