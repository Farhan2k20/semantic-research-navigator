from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_churn_data_has_target_balance():
    module = load_module(ROOT / "projects" / "churn-risk-model" / "train.py")
    df = module.make_churn_data(500)
    assert "churned" in df
    assert 0.05 < df["churned"].mean() < 0.75


def test_housing_data_has_positive_prices():
    module = load_module(ROOT / "projects" / "house-price-regression" / "train.py")
    df = module.make_housing_data(500)
    assert df["sale_price"].min() > 0
    assert {"square_feet", "neighborhood", "sale_price"}.issubset(df.columns)


def test_sentiment_dataset_has_two_classes():
    module = load_module(ROOT / "projects" / "sentiment-text-classifier" / "train.py")
    df = module.make_sentiment_data(2)
    assert set(df["positive"]) == {0, 1}
    assert df["review"].str.len().min() > 10


def test_credit_data_has_risk_features():
    module = load_module(ROOT / "projects" / "credit-risk-scorer" / "train.py")
    df = module.make_credit_data(500)
    assert "defaulted" in df
    assert df["credit_score"].between(300, 850).all()
