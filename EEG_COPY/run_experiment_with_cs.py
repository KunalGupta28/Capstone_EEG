"""
run_experiment_with_cs.py
=========================
Runs TWO experiments per dataset x subject x model:
  Experiment A  ("Full")  — all channels (baseline)
  Experiment B  ("CS")    — LS-BJOA selected channels

Outputs:
  results/channel_selection_comparison.csv
  results/channel_selection/{dataset}_{subject}_*.npy / *.png
"""

import os
import sys
import csv
import time
import shutil
import warnings
import traceback

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — add BOTH project roots so we can import everything
# ---------------------------------------------------------------------------
COPY_ROOT = os.path.dirname(os.path.abspath(__file__))          # EEG_COPY/
PROJECT_ROOT = os.path.dirname(COPY_ROOT)                       # EEG/

for p in [COPY_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Imports from the existing pipeline (EEG/) -----------------------------
from models.models    import MODEL_REGISTRY
from pipeline.utils   import set_seed, get_device, load_subject_data
from pipeline.train   import train_and_evaluate_subject
from pipeline.eval    import print_metrics_table

# -- Imports from the new channel-selection modules (EEG_COPY/) -------------
from channel_selection.jaya_optimizer import run_ls_bjoa
from channel_selection.channel_mask  import apply_mask, get_candidate_indices, get_min_channels
from visualization.topomap           import plot_channel_selection_frequency, plot_fitness_history


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

DATASETS = {
    "BCI-4-2a": {
        "dir": os.path.join(PROJECT_ROOT, "processed_data", "BCI-4-2a-preprocessed"),
        "is_binary": False,
        "num_classes": 4,
    },
    "BCI_IV_1": {
        "dir": os.path.join(PROJECT_ROOT, "processed_data", "BCI_IV_1_mat-preprocessed"),
        "is_binary": True,
        "num_classes": 2,
    },
    "BCI_III_IVa": {
        "dir": os.path.join(PROJECT_ROOT, "processed_data", "BCI-III-IVa-preprocessed"),
        "is_binary": True,
        "num_classes": 2,
    },
}

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CS_RESULTS_DIR = os.path.join(RESULTS_DIR, "channel_selection")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def save_cs_data_to_temp(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    subject_id: str,
    temp_dir: str,
) -> None:
    """
    Save channel-selected arrays using the _X_train/_X_test naming
    convention so that the existing load_subject_data() can load them.
    """
    os.makedirs(temp_dir, exist_ok=True)
    np.save(os.path.join(temp_dir, f"{subject_id}_X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(temp_dir, f"{subject_id}_X_test.npy"),  X_test.astype(np.float32))
    np.save(os.path.join(temp_dir, f"{subject_id}_y_train.npy"), y_train.astype(np.int64))
    np.save(os.path.join(temp_dir, f"{subject_id}_y_test.npy"),  y_test.astype(np.int64))


def save_comparison_csv(rows: list, save_path: str) -> None:
    """Write the comparison table as CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = [
        "dataset", "subject", "model", "experiment",
        "channels", "accuracy", "precision", "recall",
        "f1", "roc_auc", "train_time_s",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[main] Comparison CSV saved -> {save_path}")


def print_comparison_row(
    dataset_name: str,
    subject_id: str,
    model_name: str,
    exp_label: str,
    n_channels: int,
    metrics: dict,
    selected_channels: list = None,
    fitness_history: list = None,
) -> None:
    """Pretty-print one result row."""
    acc   = metrics.get("accuracy",  0)
    f1    = metrics.get("f1",        0)
    auc   = metrics.get("roc_auc",   "N/A")
    auc_s = f"{auc:.4f}" if isinstance(auc, float) else auc

    print(f"[{exp_label:4s}]  Channels: {n_channels:3d} | "
          f"Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc_s}")

    if selected_channels is not None:
        print(f"Selected channels: {selected_channels}")
    if fitness_history is not None:
        hist_str = ", ".join(f"{v:.3f}" for v in fitness_history[:10])
        if len(fitness_history) > 10:
            hist_str += ", ..."
        print(f"Fitness history: [{hist_str}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  LS-BJOA Channel Selection + Deep Learning Pipeline")
    print("=" * 70)

    set_seed(CONFIG["seed"])
    device = get_device()

    all_csv_rows = []
    total_start = time.time()

    for ds_name, ds_cfg in DATASETS.items():
        processed_dir = ds_cfg["dir"]
        subjects = get_subjects_from_dir(processed_dir)

        if not subjects:
            print(f"\n[main] WARNING -- no data found for {ds_name} in {processed_dir}, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}  |  Subjects: {len(subjects)}")
        print(f"{'='*70}")

        # Collect selected indices per subject for the frequency bar chart
        all_selected_indices = []
        ds_n_channels = None

        for subject_id in subjects:
            print(f"\n{'~'*60}")
            print(f"  {ds_name} | Subject: {subject_id}")
            print(f"{'~'*60}")

            # ---------------------------------------------------------------
            # 1. Load original data
            # ---------------------------------------------------------------
            try:
                X_train, X_test, y_train, y_test, num_classes = load_subject_data(
                    processed_dir, subject_id
                )
            except Exception as exc:
                print(f"[main] ERROR loading {subject_id}: {exc}")
                traceback.print_exc()
                continue

            n_channels_full = X_train.shape[1]
            if ds_n_channels is None:
                ds_n_channels = n_channels_full

            # ---------------------------------------------------------------
            # 2. Experiment A — Full channels
            # ---------------------------------------------------------------
            print(f"\n  >>> Experiment A: FULL channels ({n_channels_full} ch)")
            full_start = time.time()
            try:
                full_results = train_and_evaluate_subject(
                    subject_id=subject_id,
                    processed_dir=processed_dir,
                    results_dir=os.path.join(RESULTS_DIR, "full", ds_name),
                    config=CONFIG,
                    model_registry=MODEL_REGISTRY,
                    device=device,
                )
            except Exception as exc:
                print(f"[main] ERROR in full experiment for {subject_id}: {exc}")
                traceback.print_exc()
                full_results = {}
            full_time = time.time() - full_start

            # ---------------------------------------------------------------
            # 3. Run LS-BJOA channel selection (on training data only)
            # ---------------------------------------------------------------
            print(f"\n  >>> Running LS-BJOA channel selection ...")
            try:
                cs_result = run_ls_bjoa(
                    X_train=X_train,
                    y_train=y_train,
                    n_channels=n_channels_full,
                    candidate_indices=get_candidate_indices(ds_name),
                    min_channels=get_min_channels(ds_name),
                    n_pop=10,
                    n_iter=20,
                    seed=CONFIG["seed"],
                )
                selected_indices = cs_result["selected_indices"]
                n_selected = cs_result["n_selected"]
                fitness_history = cs_result["fitness_history"]

                print(f"  LS-BJOA done: {n_selected}/{n_channels_full} channels selected")
                print(f"  Best fitness: {cs_result['best_fitness']:.4f}")
                print(f"  Selected indices: {selected_indices}")

                all_selected_indices.append(selected_indices)

                # Save CS artifacts
                os.makedirs(CS_RESULTS_DIR, exist_ok=True)
                np.save(
                    os.path.join(CS_RESULTS_DIR, f"{ds_name}_{subject_id}_selected_channels.npy"),
                    np.array(selected_indices),
                )
                np.save(
                    os.path.join(CS_RESULTS_DIR, f"{ds_name}_{subject_id}_fitness_history.npy"),
                    np.array(fitness_history),
                )

                # Fitness curve plot
                plot_fitness_history(
                    fitness_history=fitness_history,
                    subject_id=subject_id,
                    dataset_name=ds_name,
                    save_path=os.path.join(
                        CS_RESULTS_DIR, f"{ds_name}_{subject_id}_fitness_curve.png"
                    ),
                )

            except Exception as exc:
                print(f"[main] WARNING -- LS-BJOA failed for {subject_id}: {exc}")
                traceback.print_exc()
                print("[main] Falling back to all channels for CS experiment.")
                selected_indices = list(range(n_channels_full))
                n_selected = n_channels_full
                fitness_history = []

            # ---------------------------------------------------------------
            # 4. Apply channel mask and run Experiment B
            # ---------------------------------------------------------------
            X_train_cs = apply_mask(X_train, selected_indices)
            X_test_cs  = apply_mask(X_test,  selected_indices)

            # Save to temp dir so load_subject_data can find it
            temp_dir = os.path.join(COPY_ROOT, "_temp_cs_data")
            save_cs_data_to_temp(X_train_cs, X_test_cs, y_train, y_test, subject_id, temp_dir)

            print(f"\n  >>> Experiment B: CS channels ({n_selected} ch)")
            cs_start = time.time()
            try:
                cs_results = train_and_evaluate_subject(
                    subject_id=subject_id,
                    processed_dir=temp_dir,
                    results_dir=os.path.join(RESULTS_DIR, "cs", ds_name),
                    config=CONFIG,
                    model_registry=MODEL_REGISTRY,
                    device=device,
                )
            except Exception as exc:
                print(f"[main] ERROR in CS experiment for {subject_id}: {exc}")
                traceback.print_exc()
                cs_results = {}
            cs_time = time.time() - cs_start

            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            # ---------------------------------------------------------------
            # 5. Print side-by-side comparison
            # ---------------------------------------------------------------
            for model_name in MODEL_REGISTRY:
                print(f"\n=== {ds_name} | Subject: {subject_id} | Model: {model_name} ===")

                if model_name in full_results:
                    print_comparison_row(
                        ds_name, subject_id, model_name, "FULL",
                        n_channels_full, full_results[model_name],
                    )
                    per_model_time = full_time / max(len(full_results), 1)
                    all_csv_rows.append({
                        "dataset": ds_name, "subject": subject_id,
                        "model": model_name, "experiment": "Full",
                        "channels": n_channels_full,
                        "accuracy":  round(full_results[model_name].get("accuracy", 0), 6),
                        "precision": round(full_results[model_name].get("precision", 0), 6)
                                     if "precision" in full_results[model_name] else "N/A",
                        "recall":    round(full_results[model_name].get("recall", 0), 6)
                                     if "recall" in full_results[model_name] else "N/A",
                        "f1":        round(full_results[model_name].get("f1", 0), 6),
                        "roc_auc":   round(full_results[model_name].get("roc_auc", 0), 6)
                                     if "roc_auc" in full_results[model_name] else "N/A",
                        "train_time_s": round(per_model_time, 1),
                    })

                if model_name in cs_results:
                    print_comparison_row(
                        ds_name, subject_id, model_name, "CS",
                        n_selected, cs_results[model_name],
                        selected_channels=selected_indices,
                        fitness_history=fitness_history,
                    )
                    per_model_time = cs_time / max(len(cs_results), 1)
                    all_csv_rows.append({
                        "dataset": ds_name, "subject": subject_id,
                        "model": model_name, "experiment": "CS",
                        "channels": n_selected,
                        "accuracy":  round(cs_results[model_name].get("accuracy", 0), 6),
                        "precision": round(cs_results[model_name].get("precision", 0), 6)
                                     if "precision" in cs_results[model_name] else "N/A",
                        "recall":    round(cs_results[model_name].get("recall", 0), 6)
                                     if "recall" in cs_results[model_name] else "N/A",
                        "f1":        round(cs_results[model_name].get("f1", 0), 6),
                        "roc_auc":   round(cs_results[model_name].get("roc_auc", 0), 6)
                                     if "roc_auc" in cs_results[model_name] else "N/A",
                        "train_time_s": round(per_model_time, 1),
                    })

        # -- Per-dataset channel frequency bar chart -----------------------
        if all_selected_indices and ds_n_channels is not None:
            plot_channel_selection_frequency(
                selected_indices_list=all_selected_indices,
                n_channels=ds_n_channels,
                dataset_name=ds_name,
                save_path=os.path.join(CS_RESULTS_DIR, f"{ds_name}_cs_bar_chart.png"),
            )

    # -- Save final CSV ----------------------------------------------------
    if all_csv_rows:
        csv_path = os.path.join(RESULTS_DIR, "channel_selection_comparison.csv")
        save_comparison_csv(all_csv_rows, csv_path)

    total_elapsed = time.time() - total_start
    print(f"\n[main] Total runtime: {total_elapsed / 60:.1f} minutes")
    print("[main] Pipeline complete.")


if __name__ == "__main__":
    main()
