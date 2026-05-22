# Credit Risk Scorer

Predicts loan default risk from borrower, loan, and payment history features.

## Why This Matters

Lenders need calibrated risk estimates to approve loans, set review thresholds, and prioritize manual underwriting.

## ML Concepts

- Imbalanced binary classification
- Gradient boosting
- Probability-based decision thresholds
- Recall and ROC-AUC evaluation

## Run

```powershell
python projects/credit-risk-scorer/train.py
```

Outputs:

- `artifacts/credit_risk_model.joblib`
- `artifacts/metrics.json`
