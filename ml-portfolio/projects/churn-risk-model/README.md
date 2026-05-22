# Customer Churn Risk Model

Predicts whether a subscription customer is likely to churn using behavioral, billing, and support features.

## Why This Matters

Retention teams need a prioritized list of customers to contact before cancellation. This model produces churn probabilities that can feed outreach campaigns or account health dashboards.

## ML Concepts

- Binary classification
- Feature preprocessing with `ColumnTransformer`
- Logistic regression as an interpretable baseline
- ROC-AUC and recall-focused evaluation

## Run

```powershell
python projects/churn-risk-model/train.py
```

Outputs:

- `artifacts/churn_model.joblib`
- `artifacts/metrics.json`
