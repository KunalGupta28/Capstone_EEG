"""
pipeline/utils.py
=================
Shared utility functions for the EEG deep learning pipeline.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set Python / NumPy / PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[utils] Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_subject_data(processed_dir: str, subject_id: str):
    """
    Load preprocessed NumPy arrays for a single subject.

    Expected file naming:
        {subject_id}_X.npy (for BCI-2a) OR {subject_id}_X_train.npy (for BCI-1)

    Returns
    -------
    X_train : np.ndarray  shape (N_train, C, T)  float32
    X_test  : np.ndarray  shape (N_test,  C, T)  float32
    y_train : np.ndarray  shape (N_train,)       long
    y_test  : np.ndarray  shape (N_test,)        long
    num_classes : int
    """
    path_X_2a = os.path.join(processed_dir, f"{subject_id}_X.npy")
    path_X_1  = os.path.join(processed_dir, f"{subject_id}_X_train.npy")

    if os.path.exists(path_X_2a):
        # BCI-IV-2a format
        X = np.load(path_X_2a).astype(np.float32)
        y = np.load(os.path.join(processed_dir, f"{subject_id}_y.npy"))
        
        # BCI-IV 2a labels might be 1-4, make them 0-3 for PyTorch CrossEntropy
        if y.min() == 1:
            y = y - 1
            
        y = y.astype(np.int64)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    elif os.path.exists(path_X_1):
        # BCI-IV-1 format
        X_train = np.load(path_X_1).astype(np.float32)
        X_test  = np.load(os.path.join(processed_dir, f"{subject_id}_X_test.npy")).astype(np.float32)
        y_train = np.load(os.path.join(processed_dir, f"{subject_id}_y_train.npy")).astype(np.int64)
        y_test  = np.load(os.path.join(processed_dir, f"{subject_id}_y_test.npy")).astype(np.int64)
    else:
        raise FileNotFoundError(f"[utils] Preprocessed data not found for subject: {subject_id}")

    num_classes = len(np.unique(y_train))

    print(f"[utils] Loaded {subject_id}: "
          f"X_train={X_train.shape}, X_test={X_test.shape}, "
          f"y_train={y_train.shape}, y_test={y_test.shape}, num_classes={num_classes}")
    return X_train, X_test, y_train, y_test, num_classes


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class EEGDataset(Dataset):
    """Minimal PyTorch Dataset wrapper for (X, y) NumPy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)          # (N, C, T)
        self.y = torch.from_numpy(y).long()   # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Data Augmentation (training-only)
# ---------------------------------------------------------------------------
def augment_eeg(
    X: np.ndarray,
    y: np.ndarray,
    noise_std: float = 0.1,
    n_noise_copies: int = 2,
    max_shift: int = 20,
    n_shift_copies: int = 2,
) -> tuple:
    """
    Generate augmented copies of EEG trials via noise injection and
    temporal jitter.  Apply ONLY to training data.

    Parameters
    ----------
    X : (n_trials, n_channels, n_times) float32
    y : (n_trials,) int64
    noise_std       : std of additive Gaussian noise
    n_noise_copies  : how many noisy copies to create
    max_shift       : maximum circular shift in samples (both directions)
    n_shift_copies  : how many shifted copies to create

    Returns
    -------
    X_aug, y_aug : augmented arrays (original data is included)
    """
    aug_X = [X]
    aug_y = [y]

    for _ in range(n_noise_copies):
        noise = np.random.randn(*X.shape).astype(np.float32) * noise_std
        aug_X.append(X + noise)
        aug_y.append(y)

    for _ in range(n_shift_copies):
        shift = np.random.randint(-max_shift, max_shift + 1)
        aug_X.append(np.roll(X, shift, axis=2))
        aug_y.append(y)

    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


def sliding_window_expand(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 200,
    stride: int = 50,
) -> tuple:
    """
    Extract overlapping sub-windows from each trial.
    Apply ONLY to training data.

    Parameters
    ----------
    X : (n_trials, n_channels, n_times) float32
    y : (n_trials,) int64
    window_size : length of each sub-window in samples
    stride      : step size between consecutive windows

    Returns
    -------
    X_win, y_win : expanded arrays
    """
    n_trials, n_ch, n_times = X.shape
    if n_times <= window_size:
        return X, y

    windows_per_trial = (n_times - window_size) // stride + 1
    total = n_trials * windows_per_trial
    X_out = np.zeros((total, n_ch, window_size), dtype=X.dtype)
    y_out = np.zeros(total, dtype=y.dtype)

    idx = 0
    for i in range(n_trials):
        for start in range(0, n_times - window_size + 1, stride):
            X_out[idx] = X[i, :, start:start + window_size]
            y_out[idx] = y[i]
            idx += 1

    return X_out[:idx], y_out[:idx]

