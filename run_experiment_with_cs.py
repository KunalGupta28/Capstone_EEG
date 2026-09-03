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

from sklearn.exceptions import ConvergenceWarning
# Suppress only sklearn SVM convergence warnings — not all warnings (Issue 11)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    "n_folds":                  3,
    "epochs":                   50,
    "batch_size":               32,
    "lr":                       5e-4,
    "weight_decay":             1e-4,
    "early_stopping_patience":  10,
    "lr_scheduler_patience":    5,
    "seed":                     42,
    "dropout":                  0.5,
}

DATASETS = {
    "BCI-4-2a": {
        "dir": os.path.join(PROJECT_ROOT, "processed_data", "BCI-4-2a-preprocessed"),
        "is_binary": False,
        "num_classes": 4,
        "sampling_rate": 250.0,
    },
    "BCI_IV_1": {
        "dir": os.path.join(PROJECT_ROOT, "processed_data", "BCI_IV_1_mat-preprocessed"),
        "is_binary": True,
        "num_classes": 2,
        "sampling_rate": 100.0,
    },
    "BCI_III_IVa": {
        "dir": os.path.join(PROJECT_ROOT, "processed_data", "BCI-III-IVa-preprocessed"),
        "is_binary": True,
        "num_classes": 2,
        "sampling_rate": 100.0,
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
    """Write the detailed per-subject comparison table as CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = [
        "dataset", "subject", "model", "experiment",
        "channels", "crr", "accuracy", "precision", "recall",
        "f1", "kappa", "roc_auc", "train_time_s",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[main] Detailed CSV saved -> {save_path}")


def save_paper_style_csv(rows: list, save_dir: str, datasets_cfg: dict) -> None:
    """
    Generate paper-style summary CSVs (one per dataset).
    
    Each CSV has Full and Selected columns side-by-side, with results
    averaged across all subjects, matching the MAIN_PAPER.pdf table format.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Group rows by dataset
    ds_rows = {}
    for row in rows:
        ds = row["dataset"]
        if ds not in ds_rows:
            ds_rows[ds] = []
        ds_rows[ds].append(row)
    
    for ds_name, ds_data in ds_rows.items():
        is_binary = datasets_cfg.get(ds_name, {}).get("is_binary", False)
        
        # Collect per-model averages for Full and CS
        model_full = {}  # {model_name: {metric: [values]}}
        model_cs = {}
        
        for row in ds_data:
            model = row["model"]
            exp = row["experiment"]
            target = model_full if exp == "Full" else model_cs
            
            if model not in target:
                target[model] = {"accuracy": [], "f1": [], "kappa": [],
                                 "roc_auc": [], "channels": []}
            
            target[model]["accuracy"].append(float(row.get("accuracy", 0)))
            target[model]["f1"].append(float(row.get("f1", 0)))
            kappa_val = row.get("kappa", 0)
            target[model]["kappa"].append(float(kappa_val) if kappa_val != "N/A" else 0.0)
            auc_val = row.get("roc_auc", 0)
            target[model]["roc_auc"].append(float(auc_val) if auc_val != "N/A" else 0.0)
            target[model]["channels"].append(int(row.get("channels", 0)))
        
        # Compute averages
        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        # Determine total channels from the Full experiment
        total_ch = 0
        for model, vals in model_full.items():
            if vals["channels"]:
                total_ch = int(_avg(vals["channels"]))
                break
        
        # Build CSV rows
        if is_binary:
            fieldnames = [
                "Model",
                "Full Acc(%)", "Full F1-Score", "Full Kappa", "Full ROC-AUC",
                "CS Acc(%)", "CS F1-Score", "CS Kappa", "CS ROC-AUC", "Active Ch.",
            ]
        else:
            fieldnames = [
                "Model",
                "Full Acc(%)", "Full F1-Score", "Full Kappa",
                "CS Acc(%)", "CS F1-Score", "CS Kappa", "Active Ch.",
            ]
        
        csv_rows = []
        all_models = list(dict.fromkeys(
            [r["model"] for r in ds_data]
        ))  # preserve insertion order
        
        for model in all_models:
            full = model_full.get(model, {})
            cs = model_cs.get(model, {})
            
            full_acc = _avg(full.get("accuracy", [])) * 100
            full_f1 = _avg(full.get("f1", []))
            full_kappa = _avg(full.get("kappa", []))
            full_auc = _avg(full.get("roc_auc", []))
            
            cs_acc = _avg(cs.get("accuracy", [])) * 100
            cs_f1 = _avg(cs.get("f1", []))
            cs_kappa = _avg(cs.get("kappa", []))
            cs_auc = _avg(cs.get("roc_auc", []))
            active_ch = int(_avg(cs.get("channels", []))) if cs.get("channels") else 0
            
            if is_binary:
                csv_rows.append({
                    "Model": model,
                    "Full Acc(%)": round(full_acc, 2),
                    "Full F1-Score": round(full_f1, 3),
                    "Full Kappa": round(full_kappa, 3),
                    "Full ROC-AUC": round(full_auc, 3),
                    "CS Acc(%)": round(cs_acc, 2),
                    "CS F1-Score": round(cs_f1, 3),
                    "CS Kappa": round(cs_kappa, 3),
                    "CS ROC-AUC": round(cs_auc, 3),
                    "Active Ch.": active_ch,
                })
            else:
                csv_rows.append({
                    "Model": model,
                    "Full Acc(%)": round(full_acc, 2),
                    "Full F1-Score": round(full_f1, 3),
                    "Full Kappa": round(full_kappa, 3),
                    "CS Acc(%)": round(cs_acc, 2),
                    "CS F1-Score": round(cs_f1, 3),
                    "CS Kappa": round(cs_kappa, 3),
                    "Active Ch.": active_ch,
                })
        
        csv_path = os.path.join(save_dir, f"{ds_name}_paper_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        # Also print the table to console
        print(f"\n{'='*80}")
        print(f"  Paper-Style Summary: {ds_name} "
              f"(Full: {total_ch} ch)")
        print(f"{'='*80}")
        
        # Print header
        if is_binary:
            print(f"{'Model':<15} | {'Full Acc%':>9} {'Full F1':>8} {'Full κ':>7} {'Full AUC':>9} | "
                  f"{'CS Acc%':>8} {'CS F1':>7} {'CS κ':>6} {'CS AUC':>8} {'Act.Ch':>7}")
        else:
            print(f"{'Model':<15} | {'Full Acc%':>9} {'Full F1':>8} {'Full κ':>7} | "
                  f"{'CS Acc%':>8} {'CS F1':>7} {'CS κ':>6} {'Act.Ch':>7}")
        print("-" * 80)
        
        for r in csv_rows:
            if is_binary:
                print(f"{r['Model']:<15} | {r['Full Acc(%)']:>9.2f} {r['Full F1-Score']:>8.3f} "
                      f"{r['Full Kappa']:>7.3f} {r['Full ROC-AUC']:>9.3f} | "
                      f"{r['CS Acc(%)']:>8.2f} {r['CS F1-Score']:>7.3f} "
                      f"{r['CS Kappa']:>6.3f} {r['CS ROC-AUC']:>8.3f} {r['Active Ch.']:>7}")
            else:
                print(f"{r['Model']:<15} | {r['Full Acc(%)']:>9.2f} {r['Full F1-Score']:>8.3f} "
                      f"{r['Full Kappa']:>7.3f} | "
                      f"{r['CS Acc(%)']:>8.2f} {r['CS F1-Score']:>7.3f} "
                      f"{r['CS Kappa']:>6.3f} {r['Active Ch.']:>7}")
        
        print(f"\n[main] Paper summary CSV saved -> {csv_path}")


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
    kappa = metrics.get("kappa",     "N/A")
    auc   = metrics.get("roc_auc",   "N/A")
    
    auc_s   = f"{auc:.4f}" if isinstance(auc, float) else auc
    kappa_s = f"{kappa:.4f}" if isinstance(kappa, float) else kappa

    print(f"[{exp_label:4s}]  Channels: {n_channels:3d} | "
          f"Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa_s} | ROC-AUC: {auc_s}")

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
    print("=" * 70, flush=True)
    print("  LS-BJOA Channel Selection + Deep Learning Pipeline", flush=True)
    print("=" * 70, flush=True)

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
            print(f"\n{'~'*60}", flush=True)
            print(f"  {ds_name} | Subject: {subject_id}", flush=True)
            print(f"{'~'*60}", flush=True)

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
            print(f"\n  >>> Experiment A: FULL channels ({n_channels_full} ch)", flush=True)
            full_start = time.time()
            try:
                full_results = train_and_evaluate_subject(
                    subject_id=subject_id,
                    processed_dir=processed_dir,
                    results_dir=os.path.join(RESULTS_DIR, "full", ds_name),
                    config=CONFIG,
                    model_registry=MODEL_REGISTRY,
                    device=device,
                    sampling_rate=ds_cfg.get("sampling_rate", 100.0),
                )
            except Exception as exc:
                print(f"[main] ERROR in full experiment for {subject_id}: {exc}")
                traceback.print_exc()
                full_results = {}
            full_time = time.time() - full_start

            # ---------------------------------------------------------------
            # 3. Run LS-BJOA channel selection (on training data only)
            # ---------------------------------------------------------------
            print(f"\n  >>> Running LS-BJOA channel selection ...", flush=True)
            try:
                cs_result = run_ls_bjoa(
                    X_train=X_train,
                    y_train=y_train,
                    n_channels=n_channels_full,
                    candidate_indices=get_candidate_indices(ds_name),
                    min_channels=get_min_channels(ds_name),
                    is_binary=ds_cfg["is_binary"],
                    seed=CONFIG["seed"],
                    sampling_rate=ds_cfg.get("sampling_rate", 100.0),
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
            temp_dir = os.path.join(PROJECT_ROOT, "_temp_cs_data")
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
                    sampling_rate=ds_cfg.get("sampling_rate", 100.0),
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
                        "crr": 0.0,
                        "accuracy":  round(full_results[model_name].get("accuracy", 0), 6),
                        "precision": round(full_results[model_name].get("precision", 0), 6),
                        "recall":    round(full_results[model_name].get("recall", 0), 6),
                        "f1":        round(full_results[model_name].get("f1", 0), 6),
                        "kappa":     round(full_results[model_name].get("kappa", 0), 6)
                                     if "kappa" in full_results[model_name] else "N/A",
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
                    crr_val = round(1.0 - (n_selected / n_channels_full), 6) if n_channels_full else 0.0
                    all_csv_rows.append({
                        "dataset": ds_name, "subject": subject_id,
                        "model": model_name, "experiment": "CS",
                        "channels": n_selected,
                        "crr": crr_val,
                        "accuracy":  round(cs_results[model_name].get("accuracy", 0), 6),
                        "precision": round(cs_results[model_name].get("precision", 0), 6),
                        "recall":    round(cs_results[model_name].get("recall", 0), 6),
                        "f1":        round(cs_results[model_name].get("f1", 0), 6),
                        "kappa":     round(cs_results[model_name].get("kappa", 0), 6)
                                     if "kappa" in cs_results[model_name] else "N/A",
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
        
        # Paper-style summary tables (one CSV per dataset)
        save_paper_style_csv(all_csv_rows, RESULTS_DIR, DATASETS)

    total_elapsed = time.time() - total_start
    print(f"\n[main] Total runtime: {total_elapsed / 60:.1f} minutes")
    print("[main] Pipeline complete.")


if __name__ == "__main__":
    main()
