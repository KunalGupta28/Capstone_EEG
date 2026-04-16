import os
import argparse
import numpy as np
import mne
from mne.preprocessing import ICA
from sklearn.preprocessing import StandardScaler

from .config import (
    DATA_DIR, PROCESSED_DIR, FREQ_BANDS, EPOCH_TMIN, EPOCH_TMAX, 
    EEG_CHANNELS, EOG_CHANNELS
)
from .bci_iv_2a_loader import load_mat_file

def preprocess_subject(subject_id):
    """
    Preprocess a single subject's data from BCI IV 2a.
    subject_id: e.g., 'A01T'
    """
    file_path = os.path.join(DATA_DIR, f'{subject_id}.mat')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"\n--- Processing Subject: {subject_id} ---")
    
    # 1. Load Data
    raw_list, events_list = load_mat_file(file_path)
    if not raw_list:
        print(f"No valid motor imagery runs found for {subject_id}.")
        return
        
    # We concatenate the runs for better ICA performance
    raw = mne.concatenate_raws(raw_list)
    
    # MNE's concatenate_raws updates event times automatically, but we need to pass a 
    # list of events to mne.events_from_annotations or similar, or we can manually concatenate 
    # events by keeping track of the cumulated samples.
    # Actually, it's easier to filter, run ICA, then segment run-by-run.
    # Wait, concatenate_raws does not return concatenated events unless we use annotations.
    # Let's add annotations to the raws before concatenating.
    
    for i in range(len(raw_list)):
        ev = events_list[i]
        # create annotations from events (duration 0 is fine, description is the class)
        onset = ev[:, 0] / raw_list[i].info['sfreq']
        duration = np.zeros_like(onset)
        description = [str(x) for x in ev[:, 2]]
        annot = mne.Annotations(onset=onset, duration=duration, description=description)
        raw_list[i].set_annotations(annot)
        
    # Now concatenate
    raw = mne.concatenate_raws(raw_list)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # 2. Filtering (4 - 40 Hz)
    print("Applying bandpass filter (4-40Hz)...")
    raw.filter(l_freq=FREQ_BANDS['low'], h_freq=FREQ_BANDS['high'], fir_design='firwin', verbose=False)
    
    # 3. ICA for Artifact Removal
    # Fit ICA on EEG channels only
    print("Fitting ICA to remove EOG artifacts...")
    ica = ICA(n_components=min(15, len(EEG_CHANNELS)), random_state=42, max_iter='auto')
    
    # We reject segments with extreme amplitudes before fitting ICA to improve results
    reject = dict(eeg=100e-6) # 100 uV 
    ica.fit(raw, picks='eeg', reject_by_annotation=False, verbose=False) # Simplified for script
    
    # Automatically find EOG artifacts using the EOG channels
    # We have 3 EOG channels
    for eog_ch in EOG_CHANNELS:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_ch, verbose=False)
        ica.exclude.extend(eog_indices)
        
    # Remove duplicates
    ica.exclude = list(set(ica.exclude))
    print(f"ICA excluded components: {ica.exclude}")
    
    # Apply ICA
    ica.apply(raw, verbose=False)
    
    # 4. Trial Segmentation (Epoching)
    print("Segmenting trials...")
    # Picks only EEG channels for the final output
    picks = mne.pick_types(raw.info, eeg=True, eog=False)
    
    epochs = mne.Epochs(
        raw, events, event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        picks=picks, baseline=None, preload=True, verbose=False
    )
    
    # 5. Normalization
    # Get data as a numpy array: shape (n_trials, n_channels, n_times)
    X = epochs.get_data()
    y = epochs.events[:, 2] # Event IDs (classes)
    
    print(f"Epochs shape: {X.shape}")
    
    # Z-score normalization per channel (over time and trials)
    # We will reshape X to (n_trials * n_times, n_channels), scale, then reshape back
    n_trials, n_channels, n_times = X.shape
    X_reshaped = np.transpose(X, (0, 2, 1)).reshape(-1, n_channels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # Reshape back to (n_trials, n_channels, n_times)
    X_norm = np.transpose(X_scaled.reshape(n_trials, n_times, n_channels), (0, 2, 1))
    
    # 6. Save data
    out_x_path = os.path.join(PROCESSED_DIR, f"{subject_id}_X.npy")
    out_y_path = os.path.join(PROCESSED_DIR, f"{subject_id}_y.npy")
    
    np.save(out_x_path, X_norm)
    np.save(out_y_path, y)
    
    print(f"Data saved to {PROCESSED_DIR} ({subject_id}_X.npy, {subject_id}_y.npy)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='A01T', help="Subject ID (e.g., A01T)")
    parser.add_argument('--all', action='store_true', help="Process all available subject files")
    args = parser.parse_args()
    
    if args.all:
        import glob
        mat_files = glob.glob(os.path.join(DATA_DIR, '*.mat'))
        subjects = [os.path.basename(f).split('.')[0] for f in mat_files]
        for sub in subjects:
            preprocess_subject(sub)
    else:
        preprocess_subject(args.subject)
