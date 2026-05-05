"""
pipeline/eval.py
================
Metric computation and visualisation for the EEG pipeline.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray, is_binary: bool = False) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true        : 1-D integer/float array
    y_pred_logits : Array of raw model outputs (before sigmoid/softmax)
    is_binary     : bool, True if binary classification
    """
    if is_binary:
        # y_pred_logits is (batch, 1)
        y_scores = 1 / (1 + np.exp(-y_pred_logits.flatten()))
        y_pred   = (y_scores >= 0.5).astype(int)
        y_true   = y_true.astype(int)

        metrics = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5,
        }
    else:
        # y_pred_logits is (batch, num_classes)
        y_pred = np.argmax(y_pred_logits, axis=1)
        y_true = y_true.astype(int)

        metrics = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "f1":        f1_score(y_true, y_pred, average="macro"),
        }
    return metrics


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------
def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    save_path: str,
    title: str = "Training & Validation Loss",
) -> None:
    """Save a loss-curve figure to *save_path*."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2, color="#4C72B0")
    ax.plot(epochs, val_losses,   label="Val Loss",   linewidth=2,
            color="#DD8452", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss",  fontsize=12)
    ax.set_title(title,    fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)





# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
    num_classes: int = 4
) -> None:
    """Save a seaborn confusion-matrix heatmap to *save_path*."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if num_classes == 2:
        labels = ["Class 0", "Class 1"]
    else:
        labels = ["Left Hand", "Right Hand", "Feet", "Tongue"]

    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
def print_metrics_table(results_dict: dict) -> None:
    """
    Pretty-print a nested dict of results:
      { subject_id: { model_name: { metric: value } } }
    """
    header = f"{'Subject':<22} {'Model':<12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for subject, models in sorted(results_dict.items()):
        for model_name, m in sorted(models.items()):
            if "roc_auc" in m:
                # Binary metrics
                print(
                    f"{subject:<22} {model_name:<12} "
                    f"{m.get('accuracy',  0):.4f}  "
                    f"{m.get('precision', 0):.4f}  "
                    f"{m.get('recall',    0):.4f}  "
                    f"{m.get('f1',        0):.4f}  "
                    f"{m.get('roc_auc',   0):.4f}"
                )
            else:
                # Multi-class metrics
                print(
                    f"{subject:<22} {model_name:<12} "
                    f"{m.get('accuracy',  0):.4f}  "
                    f"{'N/A':>7}  {'N/A':>7}  "
                    f"{m.get('f1',        0):.4f}  "
                    f"{'N/A':>7}"
                )
    print(sep + "\n")
