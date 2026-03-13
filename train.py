from pathlib import Path

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
from utils import (
    find_best_threshold,
    evaluate_at_threshold,
    print_metrics,
    plot_series_with_incidents,
)


WINDOW_SIZE = 50
HORIZON = 15
TEST_SIZE = 0.25
RANDOM_STATE = 42

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import make_synthetic_series, make_windows, chronological_split
from utils import (
    find_best_threshold,
    evaluate_at_threshold,
    print_metrics,
    plot_series_with_incidents,
)


WINDOW_SIZE = 50
HORIZON = 15
RANDOM_STATE = 42

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

OUTPUT_DIR = Path("outputs")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) create synthetic data
    series, incident = make_synthetic_series(n_steps=6000, seed=RANDOM_STATE)

    # save quick plot for inspection
    plot_series_with_incidents(
        series,
        incident,
        save_path=OUTPUT_DIR / "synthetic_series.png",
    )

    # 2) convert to sliding-window dataset
    X, y = make_windows(
        series,
        incident,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
    )

    print("full dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("positive rate:", round(y.mean(), 4))

    # 3) chronological split
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X,
        y,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
    )

    print("\nsplit sizes")
    print("train:", X_train.shape, y_train.shape)
    print("val:  ", X_val.shape, y_val.shape)
    print("test: ", X_test.shape, y_test.shape)

    # 4) normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5) train simple baseline
    model = LogisticRegression(
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    # 6) threshold tuning on validation split
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    best_thr, table = find_best_threshold(y_val, val_probs)

    print("\nvalidation threshold sweep")
    for row in table:
        print(
            f"thr={row['threshold']:.2f}  "
            f"p={row['precision']:.3f}  "
            f"r={row['recall']:.3f}  "
            f"f1={row['f1']:.3f}"
        )

    print("\nbest validation threshold based on f1:")
    print(best_thr)

    # 7) final evaluation on test split
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_metrics = evaluate_at_threshold(
        y_test,
        test_probs,
        threshold=best_thr["threshold"],
    )
    print_metrics("test results", test_metrics, best_thr["threshold"])

    # small debug info
    print("\nfirst 10 predicted probabilities on test:")
    print(np.round(test_probs[:10], 3))

    print("\nartifacts saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
