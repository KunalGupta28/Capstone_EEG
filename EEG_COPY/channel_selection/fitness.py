import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def evaluate_fitness(
    mask: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidate_indices: list,
    min_channels: int,
    is_binary: bool = False,
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

    # Issue 6: Normalization
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)

    # Issue 3: max_iter=2000
    svm = SVC(C=1, kernel='rbf', max_iter=2000, probability=is_binary)
    
    # Issues 2 & 7: StratifiedKFold with fallback for small datasets
    n_splits = min(3, np.min(np.bincount(y_train)))
    if n_splits < 2:
        return 0.0 
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_preds = np.zeros_like(y_train)
    y_probs = np.zeros_like(y_train, dtype=float)
    
    try:
        for train_idx, val_idx in skf.split(X_flat, y_train):
            svm.fit(X_flat[train_idx], y_train[train_idx])
            y_preds[val_idx] = svm.predict(X_flat[val_idx])
            if is_binary:
                y_probs[val_idx] = svm.predict_proba(X_flat[val_idx])[:, 1]
    except Exception:
        return 0.0

    # Issue 4: Better fitness formulation
    if is_binary:
        f1 = f1_score(y_train, y_preds, average='macro', zero_division=0)
        # roc_auc_score fails if only one class in y_train, though SKF prevents this mostly
        try:
            auc = roc_auc_score(y_train, y_probs)
        except ValueError:
            auc = 0.5
        fitness = 0.7 * f1 + 0.2 * auc + 0.1 * (1.0 - channel_ratio)
    else:
        macro_f1 = f1_score(y_train, y_preds, average='macro', zero_division=0)
        acc = accuracy_score(y_train, y_preds)
        fitness = 0.7 * macro_f1 + 0.2 * acc + 0.1 * (1.0 - channel_ratio)

    return float(fitness)
