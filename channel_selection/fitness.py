"""
channel_selection/fitness.py
============================
Evaluates the quality of a candidate channel mask using a fast proxy
classifier (LinearSVC) trained on **bandpower** features.

Previous version used log-variance features, which were identically zero
after per-trial Z-score normalization (variance ≡ 1.0 → log(1) = 0).
Bandpower features (mu 8-13 Hz, beta 13-30 Hz power spectral density)
survive Z-scoring because they capture frequency-domain structure that
time-domain standardisation does not destroy.
"""

import numpy as np
from scipy.signal import welch
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score


def _extract_bandpower_features(
    X: np.ndarray,
    sampling_rate: float = 100.0,
    is_binary: bool = False,
) -> np.ndarray:
    """
    Extract bandpower features.
    Binary: mu (8-13 Hz) and beta (13-30 Hz)
    Multiclass: theta (4-8), mu (8-13), beta (13-30), gamma (30-50)
    """
    n_trials, n_channels, n_times = X.shape

    # Welch PSD — dynamic segment length for ~1Hz resolution
    nperseg = min(int(sampling_rate), n_times)
    freqs, psd = welch(X, fs=sampling_rate, nperseg=nperseg, axis=2)
    # psd shape: (n_trials, n_channels, n_freqs)

    # Frequency band masks
    mu_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)

    # Average power in each band
    mu_power = np.mean(psd[:, :, mu_mask], axis=2)      # (n_trials, n_channels)
    beta_power = np.mean(psd[:, :, beta_mask], axis=2)   # (n_trials, n_channels)

    # Log-transform for better classifier separability
    mu_power = np.log(np.clip(mu_power, 1e-10, None))
    beta_power = np.log(np.clip(beta_power, 1e-10, None))

    if not is_binary:
        theta_mask = (freqs >= 4) & (freqs < 8)
        gamma_mask = (freqs > 30) & (freqs <= 50)
        
        theta_power = np.mean(psd[:, :, theta_mask], axis=2)
        gamma_power = np.mean(psd[:, :, gamma_mask], axis=2)
        
        theta_power = np.log(np.clip(theta_power, 1e-10, None))
        gamma_power = np.log(np.clip(gamma_power, 1e-10, None))
        
        features = np.concatenate([theta_power, mu_power, beta_power, gamma_power], axis=1)
    else:
        features = np.concatenate([mu_power, beta_power], axis=1)
        
    return features


def evaluate_fitness(
    mask: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidate_indices: list,
    min_channels: int,
    is_binary: bool = False,
    cache: dict = None,
    sampling_rate: float = 100.0,
) -> float:
    """
    Computes fitness for a given channel mask using LinearSVC on bandpower
    features (mu + beta band log-power).

    Parameters
    ----------
    mask              : binary mask of shape (n_channels,)
    X_train           : raw signal array of shape (n_trials, n_channels, n_times)
    y_train           : labels of shape (n_trials,)
    candidate_indices : list of forced channel indices
    min_channels      : minimum number of channels required
    is_binary         : bool, whether classification is binary
    cache             : dict, shared fitness cache
    sampling_rate     : sampling frequency in Hz (100 for DS1/DS3, 250 for 2a)
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

    # 4. Feature Extraction: Bandpower (mu + beta for binary, +theta/gamma for multiclass)
    X_selected = X_train[:, mask_copy == 1, :]
    X_flat = _extract_bandpower_features(X_selected, sampling_rate=sampling_rate, is_binary=is_binary)

    # 5. Stratified 80/20 train/val split
    min_class_count = np.min(np.bincount(y_train.astype(int)))
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
    clf = LinearSVC(C=0.1, max_iter=2000, dual=False, random_state=42)

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
    # Formula: 0.7 * F1 + 0.2 * Acc + 0.1 * (1 - channel_ratio)
    fitness = 0.7 * macro_f1 + 0.2 * acc + 0.1 * (1.0 - channel_ratio)

    fitness = float(fitness)
    if cache is not None:
        cache[cache_key] = fitness
    return fitness
