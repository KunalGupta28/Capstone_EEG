"""
channel_selection/fitness.py
============================
Optimizations applied:
1. Log-variance feature extraction: reduces dimension from (channels * times) to (channels).
2. LinearSVC classifier instead of RBF kernel SVC (orders of magnitude faster).
3. 80/20 train/validation stratified single split instead of 3-fold CV.
4. Dict-based caching of computed mask fitnesses to skip duplicate solutions.
5. Replaced AUC with accuracy in binary fitness formula for faster estimation.
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

def evaluate_fitness(
    mask: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidate_indices: list,
    min_channels: int,
    is_binary: bool = False,
    cache: dict = None,
) -> float:
    """
    Computes fitness for a given channel mask using LinearSVC on log-variance features.
    
    Parameters
    ----------
    mask              : binary mask of shape (n_channels,)
    X_train           : raw signal array of shape (n_trials, n_channels, n_times)
    y_train           : labels of shape (n_trials,)
    candidate_indices : list of forced channel indices
    min_channels      : minimum number of channels required
    is_binary         : bool, whether classification is binary
    cache             : dict, shared fitness cache
    """
    # 1. Force candidate (motor cortex) channels
    mask_copy = mask.copy()
    for idx in candidate_indices:
        mask_copy[idx] = 1

    # 2. Check cache
    cache_key = tuple(mask_copy.astype(int))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # 3. Enforce channel count constraint
    selected_channels = int(np.sum(mask_copy))
    if selected_channels < min_channels:
        if cache is not None:
            cache[cache_key] = 0.0
        return 0.0

    total_channels = len(mask_copy)
    channel_ratio = selected_channels / total_channels

    # 4. Feature Extraction: Log-Variance
    X_selected = X_train[:, mask_copy == 1, :]
    variances = np.var(X_selected, axis=2)
    X_flat = np.log(np.clip(variances, 1e-10, None))  # Shape: (n_trials, n_selected_channels)

    # 5. Stratified 80/20 train/val split (using 5-fold StratifiedKFold first split)
    min_class_count = np.min(np.bincount(y_train))
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        if cache is not None:
            cache[cache_key] = 0.0
        return 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X_flat, y_train))

    # 6. Scaling
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_flat[train_idx])
        X_val_scaled = scaler.transform(X_flat[val_idx])
    except Exception:
        if cache is not None:
            cache[cache_key] = 0.0
        return 0.0

    # 7. Fast Linear SVM Proxy Classifier
    clf = LinearSVC(C=0.1, max_iter=500, dual=False, random_state=42)
    
    try:
        clf.fit(X_train_scaled, y_train[train_idx])
        y_pred = clf.predict(X_val_scaled)
    except Exception:
        if cache is not None:
            cache[cache_key] = 0.0
        return 0.0

    # 8. Compute metrics on validation split
    y_val_true = y_train[val_idx]
    acc = accuracy_score(y_val_true, y_pred)
    macro_f1 = f1_score(y_val_true, y_pred, average='macro', zero_division=0)

    # 9. Compute multi-objective fitness
    # Formula uses 0.7 * F1 + 0.2 * Acc + 0.1 * (1 - channel_ratio)
    # This aligns accuracy-dominant preference (w1=0.9) and channel-reduction (w2=0.1)
    fitness = 0.7 * macro_f1 + 0.2 * acc + 0.1 * (1.0 - channel_ratio)

    fitness = float(fitness)
    if cache is not None:
        cache[cache_key] = fitness
    return fitness
