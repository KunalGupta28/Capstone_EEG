"""
pipeline/train.py
=================
Training loop, cross-validation, and subject-level orchestration.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from pipeline.utils import EEGDataset, load_subject_data, set_seed
from pipeline.eval  import (
    compute_metrics,
    plot_loss_curves,
    save_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool = False,
) -> float:
    """Run one training epoch. Returns mean batch loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        if is_binary:
            loss = criterion(logits, y_batch.float().unsqueeze(1))
        else:
            loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool = False,
) -> float:
    """Run one validation pass. Returns mean batch loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits  = model(X_batch)
            if is_binary:
                loss = criterion(logits, y_batch.float().unsqueeze(1))
            else:
                loss = criterion(logits, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Collect raw logits (numpy) for an entire DataLoader."""
    model.eval()
    all_logits = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


# ---------------------------------------------------------------------------
# 5-Fold Cross-Validation
# ---------------------------------------------------------------------------
def run_cross_validation(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    device: torch.device,
    is_binary: bool = False,
):
    """
    5-fold Stratified Cross-Validation.

    Returns
    -------
    fold_metrics   : list of metric dicts (one per fold)
    best_state     : state_dict of the best model across all folds
    fold_histories : list of (train_losses, val_losses) per fold
    """
    n_folds   = config["n_folds"]
    epochs    = config["epochs"]
    batch_sz  = config["batch_size"]
    lr        = config["lr"]
    wd        = config["weight_decay"]
    patience  = config["early_stopping_patience"]
    sched_pat = config["lr_scheduler_patience"]
    dropout   = config["dropout"]
    seed      = config["seed"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics   = []
    fold_histories = []
    best_val_loss  = float("inf")
    best_state     = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y.astype(int)), start=1):
        print(f"    [CV] Fold {fold}/{n_folds}", flush=True)

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds  = EEGDataset(X_tr, y_tr)
        val_ds    = EEGDataset(X_val, y_val)
        train_dl  = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                               num_workers=0, pin_memory=device.type == "cuda")
        val_dl    = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False,
                               num_workers=0, pin_memory=device.type == "cuda")

        # Build model for this fold
        model = model_cls(**model_kwargs, dropout=dropout).to(device)

        if is_binary:
            # Binary loss
            criterion = nn.BCEWithLogitsLoss()
        else:
            # Multi-class loss
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=sched_pat, factor=0.5
        )

        train_losses, val_losses = [], []
        no_improve = 0
        best_fold_loss  = float("inf")
        best_fold_state = None

        for epoch in range(1, epochs + 1):
            t_loss = train_one_epoch(model, train_dl, optimizer, criterion, device, is_binary)
            v_loss = validate_one_epoch(model, val_dl,   criterion, device, is_binary)
            scheduler.step(v_loss)

            train_losses.append(t_loss)
            val_losses.append(v_loss)

            if v_loss < best_fold_loss:
                best_fold_loss  = v_loss
                best_fold_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    [CV] Early stopping at epoch {epoch}", flush=True)
                    break

        # Evaluate this fold on its val set
        model.load_state_dict(best_fold_state)
        val_logits = predict_logits(model, val_dl, device)
        fold_m = compute_metrics(y_val, val_logits, is_binary=is_binary)
        fold_metrics.append(fold_m)
        fold_histories.append((train_losses, val_losses))

        if is_binary:
            print(f"    [CV] Fold {fold} -> "
                  f"Acc={fold_m['accuracy']:.4f}  F1={fold_m['f1']:.4f}  AUC={fold_m.get('roc_auc', 0):.4f}")
        else:
            print(f"    [CV] Fold {fold} -> "
                  f"Acc={fold_m['accuracy']:.4f}  F1 (Macro)={fold_m['f1']:.4f}", flush=True)

        # Track globally best state
        if best_fold_loss < best_val_loss:
            best_val_loss = best_fold_loss
            best_state    = copy.deepcopy(best_fold_state)

    return fold_metrics, best_state, fold_histories


# ---------------------------------------------------------------------------
# Subject-level orchestrator
# ---------------------------------------------------------------------------
def train_and_evaluate_subject(
    subject_id: str,
    processed_dir: str,
    results_dir: str,
    config: dict,
    model_registry: dict,
    device: torch.device,
) -> dict:
    """
    Full pipeline for one subject × all models.

    Returns
    -------
    subject_results : { model_name: { metric: value } }
    """
    set_seed(config["seed"])

    # -- Load data ----------------------------------------------------------
    X_train, X_test, y_train, y_test, num_classes = load_subject_data(processed_dir, subject_id)
    n_channels = X_train.shape[1]
    n_times    = X_train.shape[2]

    is_binary = (num_classes == 2)
    out_features = 1 if is_binary else num_classes
    
    # -- Output directories ------------------------------------------------
    plots_dir  = os.path.join(results_dir, "plots",       subject_id)
    weights_dir = os.path.join(results_dir, "weights")
    preds_dir   = os.path.join(results_dir, "predictions")
    for d in [plots_dir, weights_dir, preds_dir]:
        os.makedirs(d, exist_ok=True)

    subject_results = {}

    for model_name, model_cls in model_registry.items():
        print(f"\n  -- {model_name} --", flush=True)

        model_kwargs = {"n_channels": n_channels, "n_times": n_times, "num_classes": out_features}

        # 5-Fold CV on training data
        fold_metrics, best_state, fold_histories = run_cross_validation(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            X=X_train,
            y=y_train,
            config=config,
            device=device,
            is_binary=is_binary,
        )

        # -- Plot loss curves for each fold --------------------------------
        for fi, (tr_l, vl_l) in enumerate(fold_histories, start=1):
            plot_loss_curves(
                tr_l, vl_l,
                save_path=os.path.join(
                    plots_dir, f"{model_name}_fold{fi}_loss.png"
                ),
                title=f"{subject_id} | {model_name} | Fold {fi} Loss",
            )

        # -- Save best weights ---------------------------------------------
        weight_path = os.path.join(weights_dir, f"{subject_id}_{model_name}_best.pth")
        if best_state is not None:
            torch.save(best_state, weight_path)

        # -- Evaluate on held-out test set ---------------------------------
        model = model_cls(**model_kwargs, dropout=config["dropout"]).to(device)
        if best_state is not None:
            model.load_state_dict(best_state)

        test_ds = EEGDataset(X_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=config["batch_size"],
                             shuffle=False, num_workers=0)

        test_logits = predict_logits(model, test_dl, device)
        if is_binary:
            test_scores = 1 / (1 + np.exp(-test_logits.flatten()))
            test_pred   = (test_scores >= 0.5).astype(int)
        else:
            test_pred   = np.argmax(test_logits, axis=1)

        test_metrics = compute_metrics(y_test, test_logits, is_binary=is_binary)
        subject_results[model_name] = test_metrics

        # -- Save predictions ----------------------------------------------
        pred_path = os.path.join(preds_dir, f"{subject_id}_{model_name}_preds.npy")
        np.save(pred_path, test_logits)


        # -- Confusion matrix ---------------------------------------------
        save_confusion_matrix(
            y_test, test_pred,
            save_path=os.path.join(plots_dir, f"{model_name}_cm.png"),
            title=f"{subject_id} | {model_name} | Confusion Matrix",
            num_classes=num_classes
        )

        # -- Average CV metrics ------------------------------------------
        avg_cv = {k: float(np.mean([fm[k] for fm in fold_metrics]))
                  for k in fold_metrics[0]}
        if is_binary:
            print(f"  [CV avg] Acc={avg_cv['accuracy']:.4f}  "
                  f"F1={avg_cv['f1']:.4f}  AUC={avg_cv.get('roc_auc', 0):.4f}")
            print(f"  [Test]   Acc={test_metrics['accuracy']:.4f}  "
                  f"F1={test_metrics['f1']:.4f}  AUC={test_metrics.get('roc_auc', 0):.4f}")
        else:
            print(f"  [CV avg] Acc={avg_cv['accuracy']:.4f}  "
                  f"F1={avg_cv['f1']:.4f}")
            print(f"  [Test]   Acc={test_metrics['accuracy']:.4f}  "
                  f"F1={test_metrics['f1']:.4f}")

    return subject_results
