"""
channel_selection/logistic_map.py
=================================
Optimizations applied:
1. Optional numba JIT compiler to run the sequential chaotic recurrence loop at native speed.
2. Direct numpy pre-allocated loop fallback.
"""

import numpy as np

try:
    from numba import jit
except ImportError:
    # A dummy decorator if numba is not installed
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True, cache=True)
def _logistic_loop(sequence, length):
    for i in range(1, length):
        sequence[i] = 4.0 * sequence[i-1] * (1.0 - sequence[i-1])
    return sequence

def generate_chaotic_sequence(length: int, c0: float = 0.8) -> np.ndarray:
    """
    Returns array of `length` chaotic values in (0,1).
    Preserves c0=0.8.
    """
    sequence = np.zeros(length, dtype=np.float64)
    if length == 0:
        return sequence
    sequence[0] = c0
    return _logistic_loop(sequence, length)
