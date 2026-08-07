import numpy as np

def apply_mask(X: np.ndarray, selected_indices: list) -> np.ndarray:
    """
    X: (n_trials, n_channels, n_times)
    Returns: (n_trials, len(selected_indices), n_times)
    """
    return X[:, selected_indices, :]

def get_candidate_indices(dataset_name: str) -> list:
    """
    Returns candidate channel indices for each dataset (forced to be selected).
    These correspond to C3, Cz, C4 equivalents.
    """
    name_lower = dataset_name.lower()
    if '2a' in name_lower:
        # Fz=0, FC3=1, FC1=2, FCz=3, FC2=4, FC4=5, C5=6, C3=7, C1=8, Cz=9, C2=10, C4=11...
        return [7, 9, 11]
    elif 'ds1' in name_lower or 'iv_1' in name_lower or 'iv1' in name_lower:
        # C3=index 26, Cz=index 28, C4=index 30 (corrected from 27, 29, 31)
        return [26, 28, 30]
    elif 'ds3' in name_lower or 'iiia' in name_lower or 'iva' in name_lower:
        # C3=index 51, Cz=index 53, C4=index 55 (corrected from 0, 1, 2)
        return [51, 53, 55]
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
