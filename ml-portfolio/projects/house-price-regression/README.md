# House Price Regression

Estimates home sale price from property, neighborhood, and market condition features.

## Why This Matters

Real estate teams use regression models for pricing guidance, comparable analysis, and valuation sanity checks.

## ML Concepts

- Regression
- Synthetic tabular data with realistic signal
- Random forest modeling
- MAE, RMSE, and R2 evaluation

## Run

```powershell
python projects/house-price-regression/train.py
```

Outputs:

- `artifacts/price_model.joblib`
- `artifacts/metrics.json`
