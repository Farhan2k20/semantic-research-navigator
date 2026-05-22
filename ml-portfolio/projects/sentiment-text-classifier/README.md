# Sentiment Text Classifier

Classifies short product feedback as positive or negative using a TF-IDF text pipeline.

## Why This Matters

Support and product teams often need to summarize large volumes of reviews or tickets. This project shows how to turn raw text into useful sentiment signals.

## ML Concepts

- Natural language processing
- TF-IDF vectorization
- Logistic regression classification
- Holdout evaluation with ROC-AUC

## Run

```powershell
python projects/sentiment-text-classifier/train.py
```

Outputs:

- `artifacts/sentiment_model.joblib`
- `artifacts/metrics.json`
