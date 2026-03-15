from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import make_synthetic_series, make_windows, chronological_split
from utils import (
    evaluate_at_threshold,
    plot_series_with_incidents,
    plot_predictions,
    save_summary,
)


WINDOW_SIZE = 50
HORIZON = 15
RANDOM_STATE = 42

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

OUTPUT_DIR = Path("outputs")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    series, incident = make_synthetic_series(n_steps=6000, seed=RANDOM_STATE)

    plot_series_with_incidents(
        series,
        incident,
        save_path=OUTPUT_DIR / "synthetic_series.png",
    )

    X, y = make_windows(
        series,
        incident,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
    )

    print("dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("positive rate:", round(float(y.mean()), 4))

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X,
        y,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
    )

    print()
    print("split sizes")
    print("train:", X_train.shape)
    print("val:  ", X_val.shape)
    print("test: ", X_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 17)

    best_thr = 0.5
    best_f1 = -1.0
    rows = []

    print()
    print("validation threshold sweep")
    for thr in thresholds:
        m = evaluate_at_threshold(y_val, val_probs, thr)

        rows.append((thr, m["precision"], m["recall"], m["f1"]))

        print(
            f"thr={thr:.2f} p={m['precision']:.3f} r={m['recall']:.3f} f1={m['f1']:.3f}"
        )

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)

    with open(OUTPUT_DIR / "val_thresholds.csv", "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall,f1\n")
        for thr, p, r, f1 in rows:
            f.write(f"{thr:.3f},{p:.6f},{r:.6f},{f1:.6f}\n")

    print()
    print("chosen threshold:", round(best_thr, 3))

    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, test_probs, best_thr)

    print()
    print("test metrics")
    print("precision:", round(float(test_metrics["precision"]), 3))
    print("recall:   ", round(float(test_metrics["recall"]), 3))
    print("f1:       ", round(float(test_metrics["f1"]), 3))
    print("pr_auc:   ", round(float(test_metrics["pr_auc"]), 3))
    print("roc_auc:  ", round(float(test_metrics["roc_auc"]), 3))
    print("confusion matrix:")
    print(test_metrics["confusion_matrix"])

    plot_predictions(
        y_test,
        test_probs,
        best_thr,
        save_path=OUTPUT_DIR / "test_predictions.png",
    )

    save_summary(
        OUTPUT_DIR / "results_summary.txt",
        WINDOW_SIZE,
        HORIZON,
        float(y.mean()),
        best_thr,
        test_metrics,
    )

    print()
    print("saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
