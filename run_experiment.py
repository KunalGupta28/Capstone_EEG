"""
run_experiment.py
=================
Main entry point for the BCI Competition IV Dataset 2a deep learning pipeline.

Usage:
    python run_experiment.py

For each subject (ds1a – ds1g) × each model (EEGNet, CNN, RNN, LSTM, CNN+RNN,
CNN+LSTM):
  1. Load preprocessed data from processed_data/
  2. Run 5-fold Stratified Cross-Validation on training split
  3. Evaluate best model on held-out test split
  4. Save weights, predictions, loss curves, ROC curve, confusion matrix
  5. Print final comparison table + save as results/comparison_table.csv
"""

import os
import sys
import csv
import time
import warnings

warnings.filterwarnings("ignore")

# Make sure the project root is on sys.path so that `models` and `pipeline`
# are importable regardless of how the script is invoked.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from models.models    import MODEL_REGISTRY
from pipeline.utils   import set_seed, get_device
from pipeline.train   import train_and_evaluate_subject
from pipeline.eval    import print_metrics_table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "n_folds":                  5,
    "epochs":                   150,
    "batch_size":               32,
    "lr":                       1e-3,
    "weight_decay":             1e-4,
    "early_stopping_patience":  15,
    "lr_scheduler_patience":    7,
    "seed":                     42,
    "dropout":                  0.5,
}



PROCESSED_DIR = os.path.join(ROOT, "processed_data", "BCI-III-IVa-preprocessed")
RESULTS_DIR   = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_comparison_table(all_results: dict, save_path: str) -> None:
    """Save the full results dict as a CSV file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rows = []
    for subject, models in sorted(all_results.items()):
        for model_name, m in sorted(models.items()):
            rows.append({
                "subject":   subject,
                "model":     model_name,
                "accuracy":  round(m.get("accuracy",  0), 6),
                "precision": round(m.get("precision", 0), 6) if "precision" in m else "N/A",
                "recall":    round(m.get("recall",    0), 6) if "recall" in m else "N/A",
                "f1":        round(m.get("f1",        0), 6),
                "roc_auc":   round(m.get("roc_auc",   0), 6) if "roc_auc" in m else "N/A",
            })

    fieldnames = ["subject", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[main] Comparison table saved → {save_path}")


def get_subjects_from_dir(processed_dir: str) -> list:
    """Dynamically find all subject IDs in the processed directory."""
    subjects = set()
    if not os.path.exists(processed_dir):
        return []
        
    for filename in os.listdir(processed_dir):
        if filename.endswith("_X_train.npy"):
            subjects.add(filename.replace("_X_train.npy", ""))
        elif filename.endswith("_X.npy"):
            subjects.add(filename.replace("_X.npy", ""))
    
    return sorted(list(subjects))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  BCI Competition III Dataset IVa — Deep Learning Pipeline")
    print("=" * 60)

    set_seed(CONFIG["seed"])
    device = get_device()

    # Dynamically find subjects in the processed directory
    subjects = get_subjects_from_dir(PROCESSED_DIR)
    if not subjects:
        raise RuntimeError(
            f"No preprocessed data found in '{PROCESSED_DIR}'. "
            "Please ensure the data is properly preprocessed and placed in the correct folder."
        )

    print(f"\n[main] Subjects to process: {subjects}")
    print(f"[main] Models to evaluate:  {list(MODEL_REGISTRY.keys())}")
    print(f"[main] Config: {CONFIG}\n")

    all_results  = {}
    total_start  = time.time()

    for subject_id in subjects:
        print(f"\n{'='*60}")
        print(f"  Subject: {subject_id}")
        print(f"{'='*60}")

        subj_start = time.time()
        try:
            subject_results = train_and_evaluate_subject(
                subject_id    = subject_id,
                processed_dir = PROCESSED_DIR,
                results_dir   = RESULTS_DIR,
                config        = CONFIG,
                model_registry= MODEL_REGISTRY,
                device        = device,
            )
            all_results[subject_id] = subject_results
            elapsed = time.time() - subj_start
            print(f"\n[main] {subject_id} done in {elapsed/60:.1f} min")
        except Exception as exc:
            print(f"[main] ERROR processing {subject_id}: {exc}")
            import traceback; traceback.print_exc()

    # -- Final summary -------------------------------------------------------
    if all_results:
        print_metrics_table(all_results)
        csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
        save_comparison_table(all_results, csv_path)

    total_elapsed = time.time() - total_start
    print(f"\n[main] Total runtime: {total_elapsed/60:.1f} minutes")
    print("[main] Pipeline complete. ✓")


if __name__ == "__main__":
    main()
