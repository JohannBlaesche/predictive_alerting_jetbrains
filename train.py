import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import make_synthetic_series, make_windows


WINDOW_SIZE = 50
HORIZON = 15
TEST_SIZE = 0.25
RANDOM_STATE = 42


def evaluate_with_threshold(model, x, y, threshold=0.5):
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= threshold).astype(int)

    p, r, f, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )

    print(f"threshold={threshold:.2f}")
    print(f"precision={p:.3f}")
    print(f"recall={r:.3f}")
    print(f"f1={f:.3f}")
    print(confusion_matrix(y, preds))
    print()


def main():
    series, incident = make_synthetic_series(n_steps=6000, seed=RANDOM_STATE)
    X, y = make_windows(series, incident, window_size=WINDOW_SIZE, horizon=HORIZON)

    print("dataset shape:", X.shape, y.shape)
    print("positive rate:", y.mean())

    # simple split for now
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=400)
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)

    print("=== default threshold report ===")
    print(classification_report(y_test, pred, zero_division=0))

    print("=== manual threshold checks ===")
    for thr in [0.3, 0.5, 0.7]:
        evaluate_with_threshold(model, X_test_scaled, y_test, threshold=thr)


if __name__ == "__main__":
    main()
