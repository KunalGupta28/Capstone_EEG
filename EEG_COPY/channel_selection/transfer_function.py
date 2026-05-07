import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid S-shaped transfer function."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def binarize(x: np.ndarray) -> np.ndarray:
    """Converts continuous position to binary channel mask."""
    s_x = sigmoid(x)
    rand_vals = np.random.rand(*x.shape)
    binary_mask = np.where(rand_vals < s_x, 0, 1)
    return binary_mask
