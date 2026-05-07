import numpy as np

def apply_mask(X: np.ndarray, selected_indices: list) -> np.ndarray:
    """
    X: (n_trials, n_channels, n_times)
    Returns: (n_trials, len(selected_indices), n_times)
    """
    return X[:, selected_indices, :]

def get_candidate_indices(dataset_name: str) -> list:
    """
    Returns candidate channel indices for each dataset.
    dataset_name should be matched case-insensitively against:
      '2a', 'bci-4-2a'        -> [7, 9, 11]
      'ds1', 'iv_1', 'iv1'    -> [27, 29, 31]
      'ds3', 'iiia', 'iva'    -> [0, 1, 2]
    """
    name_lower = dataset_name.lower()
    if '2a' in name_lower:
        return [7, 9, 11]
    elif 'ds1' in name_lower or 'iv_1' in name_lower or 'iv1' in name_lower:
        return [27, 29, 31]
    elif 'ds3' in name_lower or 'iiia' in name_lower or 'iva' in name_lower:
        return [0, 1, 2]
    return [0, 1, 2]

def get_min_channels(dataset_name: str) -> int:
    """Returns minimum channel constraint per dataset."""
    name_lower = dataset_name.lower()
    if '2a' in name_lower:
        return 8
    elif 'ds1' in name_lower or 'iv_1' in name_lower or 'iv1' in name_lower:
        return 15
    elif 'ds3' in name_lower or 'iiia' in name_lower or 'iva' in name_lower:
        return 20
    return 8
