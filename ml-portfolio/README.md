# Applied Machine Learning Portfolio

Four practical Python machine learning projects built to show job-ready skills: clean data generation, reproducible pipelines, feature engineering, model evaluation, saved artifacts, and clear business framing.

## Projects

| Project | ML concept | What it demonstrates |
| --- | --- | --- |
| `projects/churn-risk-model` | Binary classification | Customer retention modeling, ROC-AUC, precision/recall tradeoffs |
| `projects/house-price-regression` | Regression | Feature engineering, error metrics, interpretable tree ensembles |
| `projects/sentiment-text-classifier` | NLP classification | TF-IDF, text preprocessing, model evaluation |
| `projects/credit-risk-scorer` | Risk scoring | Imbalanced classification, calibrated probabilities, business thresholds |

Each project is self-contained and uses synthetic or embedded data so it runs without private datasets or external downloads.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

Run an individual project:

```powershell
python projects/churn-risk-model/train.py
python projects/house-price-regression/train.py
python projects/sentiment-text-classifier/train.py
python projects/credit-risk-scorer/train.py
```

Training creates project-local `artifacts/` folders containing metrics and serialized models.

## Portfolio Notes

These projects are intentionally compact but complete. They are designed for interviews where a hiring manager may ask:

- How did you split data and avoid leakage?
- Which metric did you optimize and why?
- How would this model be used in a business workflow?
- How would you monitor or improve it in production?

## Repository Structure

```text
src/ml_portfolio/        Shared utilities
projects/*/train.py      Runnable training scripts
projects/*/README.md     Project-specific framing and results
tests/                   Smoke tests for core project behavior
```
