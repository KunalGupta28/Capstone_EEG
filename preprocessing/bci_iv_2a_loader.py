import os
import numpy as np
import scipy.io as sio
import mne
from .config import SAMPLING_RATE, EEG_CHANNELS, EOG_CHANNELS

def load_mat_file(file_path):
    """
    Loads a BCI Competition IV-2a .mat file and returns MNE Raw objects.
    Each .mat file contains 9 runs, we extract runs with motor imagery tasks.
    Returns:
        raw_list: A list of mne.io.RawArray objects (one per valid run).
        events_list: A list of event arrays (one per valid run).
    """
    mat = sio.loadmat(file_path)
    data = mat['data']
    
    ch_names = EEG_CHANNELS + EOG_CHANNELS
    ch_types = ['eeg'] * len(EEG_CHANNELS) + ['eog'] * len(EOG_CHANNELS)
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=ch_types)
    
    raw_list = []
    events_list = []
    
    # Iterate over all 9 runs
    for i in range(data.shape[1]):
        run_data = data[0, i]
        X = run_data['X'][0, 0] # shape (samples, channels)
        y = run_data['y'][0, 0] # shape (trials, 1)
        trial = run_data['trial'][0, 0] # shape (trials, 1)
        
        # We only care about runs that have trials
        if trial.size == 0 or y.size == 0:
            continue
            
        # Nan handling: Replace NaNs with 0 (sometimes present at start/end)
        X[np.isnan(X)] = 0
            
        # Create MNE RawArray
        # MNE expects shape (channels, samples)
        raw = mne.io.RawArray(X.T, info, verbose=False)
        
        # Create Events Array
        # MNE events shape is (n_events, 3) 
        # Column 0: sample index
        # Column 1: previous event id (usually 0)
        # Column 2: event id (class label)
        n_events = trial.shape[0]
        events = np.zeros((n_events, 3), dtype=int)
        events[:, 0] = trial.flatten()
        events[:, 2] = y.flatten()
        
        raw_list.append(raw)
        events_list.append(events)
        
    return raw_list, events_list
