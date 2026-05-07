import numpy as np

def generate_chaotic_sequence(length: int, c0: float = 0.8) -> np.ndarray:
    """Returns array of `length` chaotic values in (0,1)."""
    sequence = np.zeros(length)
    if length == 0:
        return sequence
    sequence[0] = c0
    for i in range(1, length):
        sequence[i] = 4 * sequence[i-1] * (1 - sequence[i-1])
    return sequence
