import os

# --- Paths ---
# You can override these via environment variables or modify directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'BCI-4-2a')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')

# Create processed directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Preprocessing Parameters ---
SAMPLING_RATE = 250  # Hz
FREQ_BANDS = {
    'low': 4.0,   # Hz
    'high': 40.0  # Hz
}

# Epoching parameters (relative to cue onset at t=0)
# Based on typical BCI Competition IV-2a paradigms, motor imagery is from [0, 4] or [0.5, 2.5]
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0

# Channels mapping to be expected
# 22 EEG channels and 3 EOG channels
EEG_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]
EOG_CHANNELS = ['EOG-left', 'EOG-central', 'EOG-right']

# Classes map (Four class motor imagery: Left hand, Right hand, Both feet, Tongue)
CLASS_MAPPING = {
    1: 'left_hand',
    2: 'right_hand',
    3: 'feet',
    4: 'tongue'
}
