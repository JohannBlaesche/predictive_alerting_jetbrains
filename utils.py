import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


def find_best_threshold(y_true, probs, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    best = None
    rows = []

    for thr in thresholds:
        pred = (probs >= thr).astype(int)

        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        row = {
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        rows.append(row)

        if best is None or row["f1"] > best["f1"]:
            best = row

    return best, rows


def evaluate_at_threshold(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, probs),
        "roc_auc": roc_auc_score(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, pred),
    }
    return metrics


def print_metrics(title, metrics, threshold):
    print(f"\n=== {title} ===")
    print(f"threshold: {threshold:.3f}")
    print(f"precision: {metrics['precision']:.3f}")
    print(f"recall:    {metrics['recall']:.3f}")
    print(f"f1:        {metrics['f1']:.3f}")
    print(f"pr_auc:    {metrics['pr_auc']:.3f}")
    print(f"roc_auc:   {metrics['roc_auc']:.3f}")
    print("confusion matrix:")
    print(metrics["confusion_matrix"])


def plot_series_with_incidents(
    series, incident, save_path="series_plot.png", max_points=1200
):
    n = min(len(series), max_points)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series[:n], linewidth=1.2, label="signal")

    incident_idx = np.where(incident[:n] == 1)[0]
    if len(incident_idx) > 0:
        ax.scatter(
            incident_idx,
            series[:n][incident_idx],
            s=12,
            label="incident",
        )

    ax.set_title("Synthetic time series with incident points")
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
