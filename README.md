# Incident prediction baseline

This is a small baseline for predicting whether an incident will happen in the next `H` time steps based on the previous `W` time steps.

The current version uses a synthetic univariate time series with injected incident intervals.

## Setup

For each sliding window:

- input: last `W` values
- label: 1 if an incident happens in the next `H` steps, else 0

This turns the problem into binary classification.

## Model

The model is logistic regression on flattened windows.

It is not meant to be the final best model.  
It is just a simple baseline that is easy to train and explain.

## Evaluation

The data is split chronologically into:

- train
- validation
- test

A threshold is chosen on the validation split by trying several values and taking the best F1.

Reported metrics:

- precision
- recall
- F1
- PR-AUC
- ROC-AUC
- confusion matrix

## Outputs

Running the script creates these files in `outputs/`:

- `synthetic_series.png`
- `test_predictions.png`
- `val_thresholds.csv`
- `results_summary.txt`

## Limitations

This is only a simple baseline:

- synthetic data
- one metric only
- simple incident patterns
- no event-level alert evaluation

## Run

```bash
pip install -r requirements.txt
python src/train.py
```
