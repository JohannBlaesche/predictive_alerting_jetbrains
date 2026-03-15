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


def evaluate_at_threshold(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)

    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, probs),
        "roc_auc": roc_auc_score(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, pred),
    }


def plot_series_with_incidents(
    series, incident, save_path="series_plot.png", max_points=1200
):
    n = min(len(series), max_points)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series[:n], label="signal", linewidth=1.2)

    # shade incident regions so they are easier to see
    inside = False
    start = 0
    first_label = True

    for i in range(n):
        if incident[i] == 1 and not inside:
            inside = True
            start = i
        elif incident[i] == 0 and inside:
            ax.axvspan(
                start,
                i - 1,
                alpha=0.2,
                color="red",
                label="incident" if first_label else None,
            )
            inside = False
            first_label = False

    if inside:
        ax.axvspan(
            start,
            n - 1,
            alpha=0.2,
            color="red",
            label="incident" if first_label else None,
        )

    ax.set_title("Synthetic series")
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_predictions(
    y_true, probs, threshold, save_path="predictions.png", max_points=400
):
    n = min(len(y_true), max_points)
    pred = (probs >= threshold).astype(int)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, probs[:n], label="pred prob", linewidth=1.2)
    ax.plot(x, y_true[:n], label="true label", linewidth=1.2)
    ax.plot(x, pred[:n], label="pred alert", linewidth=1.2)
    ax.axhline(threshold, linestyle="--", linewidth=1.0, label=f"thr={threshold:.2f}")

    ax.set_title("Predictions on test split")
    ax.set_xlabel("window index")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def save_summary(path, window_size, horizon, pos_rate, best_thr, metrics):
    with open(path, "w", encoding="utf-8") as f:
        f.write("incident prediction summary\n")
        f.write("\n")
        f.write(f"W = {window_size}\n")
        f.write(f"H = {horizon}\n")
        f.write("\n")
        f.write("model: logistic regression on flattened windows\n")
        f.write("split: chronological train/val/test\n")
        f.write("threshold selection: best validation f1\n")
        f.write("\n")
        f.write(f"positive rate: {pos_rate:.4f}\n")
        f.write(f"chosen threshold: {best_thr:.4f}\n")
        f.write("\n")
        f.write("test metrics\n")
        f.write(f"precision: {metrics['precision']:.4f}\n")
        f.write(f"recall: {metrics['recall']:.4f}\n")
        f.write(f"f1: {metrics['f1']:.4f}\n")
        f.write(f"pr_auc: {metrics['pr_auc']:.4f}\n")
        f.write(f"roc_auc: {metrics['roc_auc']:.4f}\n")
        f.write(f"confusion_matrix:\n{metrics['confusion_matrix']}\n")
        f.write("\n")
        f.write("notes\n")
        f.write("- simple baseline only\n")
        f.write("- synthetic data is easier than a real alerting setup\n")
        f.write("- threshold changes the precision/recall trade-off\n")
