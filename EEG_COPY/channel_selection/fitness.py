import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def evaluate_fitness(
    mask: np.ndarray,       # binary vector length=n_channels
    X_train: np.ndarray,    # (n_trials, n_channels, n_times)
    y_train: np.ndarray,    # (n_trials,)
    candidate_indices: list,
    min_channels: int,
    w1: float = 0.9,
    w2: float = 0.1,
) -> float:
    mask_copy = mask.copy()
    for idx in candidate_indices:
        mask_copy[idx] = 1

    selected_channels = int(np.sum(mask_copy))
    if selected_channels < min_channels:
        return 0.0

    total_channels = len(mask_copy)
    channel_ratio = selected_channels / total_channels

    X_selected = X_train[:, mask_copy == 1, :]
    n_trials = X_selected.shape[0]
    X_flat = X_selected.reshape(n_trials, -1)

    svm = SVC(C=1, kernel='rbf', max_iter=500)
    # Using 3-fold CV for speed, wrap in try/except in case max_iter causes warning/error
    try:
        cv_scores = cross_val_score(svm, X_flat, y_train, cv=3, n_jobs=-1)
        cv_accuracy_error = 1.0 - np.mean(cv_scores)
    except Exception:
        # Fallback if something goes wrong
        return 0.0

    fitness = w1 * (1.0 - cv_accuracy_error) - w2 * channel_ratio
    return float(fitness)
