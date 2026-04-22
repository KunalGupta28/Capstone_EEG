import os
import glob
import numpy as np
import scipy.io as sio
import mne
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'BCI -III-IVa')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_ds3_subject(file_path):
    # Extract subject ID (e.g. 'aa', 'al', 'av', 'aw', 'ay') from filename 'data_set_IVa_aa.mat'
    subject_id = os.path.basename(file_path).split('.')[0].split('_')[-1]
    print(f"\n--- Processing Subject: {subject_id} ---")

    # Step 1: Load .mat file (cnt, mrk, nfo)
    mat = sio.loadmat(file_path)
    
    cnt = mat['cnt']              # (samples, channels)
    mrk = mat['mrk']
    nfo = mat['nfo']

    # Step 2: Create MNE Raw object
    fs = float(nfo['fs'][0,0][0,0])
    
    # Extract channel labels (118 channels)
    clab_array = nfo['clab'][0,0][0]
    clab = [str(c[0]) for c in clab_array]
    
    # Scale cnt from microvolts to volts for MNE (standard practice)
    cnt_v = cnt.astype(np.float64) * 1e-6 
    
    info = mne.create_info(ch_names=clab, sfreq=fs, ch_types=['eeg'] * len(clab))
    raw = mne.io.RawArray(cnt_v.T, info, verbose=False)

    # Step 3: Extract events from mrk.pos, dropping NaNs
    pos_all = mrk['pos'][0,0][0]
    y_all = mrk['y'][0,0][0]
    
    # Filter out NaNs
    valid_idx = ~np.isnan(y_all)
    pos_valid = pos_all[valid_idx]
    y_valid = y_all[valid_idx]
    
    print(f"Retained {len(y_valid)} labeled trials (dropped {np.sum(~valid_idx)} unlabeled test trials).")
    
    # Step 10: Convert labels (-1, +1 or 1, 2) -> (0, 1) early to form standard event IDs
    # BCI-III-IVa has labels 1 and 2
    y_mapped = np.where(y_valid == 1, 0, y_valid)
    y_mapped = np.where(y_mapped == 2, 1, y_mapped).astype(int)
    
    n_events = len(pos_valid)
    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = pos_valid
    events[:, 2] = y_mapped

    # Step 4: High-pass filter (~1 Hz)
    print("Applying high-pass filter (1 Hz)...")
    raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)

    # Step 5: (Optional) ICA without EOG OR skip
    # (Skipped)
    
    # Step 6: Bandpass filter (4–40 Hz)
    print("Applying bandpass filter (4-40 Hz)...")
    raw.filter(l_freq=4.0, h_freq=40.0, fir_design='firwin', verbose=False)

    # Step 11: Apply channel selection
    # Keep all 118 channels as requested
    picks = mne.pick_types(raw.info, eeg=True)

    # Step 7: Epoching (0.5–3.5 sec)
    print("Segmenting trials (0.5 to 3.5 sec)...")
    tmin, tmax = 0.5, 3.5
    
    # Step 8: Baseline correction
    # Baseline correction is handled during epoching natively (None, None) -> mean of epoch
    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=tmin, tmax=tmax,
        picks=picks, baseline=(None, None), preload=True, verbose=False
    )

    # Step 12: Convert to (Trials, Channels, Time)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    print(f"Extracted shape: {X.shape}")

    # Step 9: Normalize (per trial/channel)
    print("Applying normalization (Z-score)...")
    n_trials, n_channels, n_times = X.shape
    X_norm = np.zeros_like(X)
    
    for i in range(n_trials):
        for c in range(n_channels):
            mean_val = np.mean(X[i, c, :])
            std_val = np.std(X[i, c, :])
            if std_val > 0:
                X_norm[i, c, :] = (X[i, c, :] - mean_val) / std_val
            else:
                X_norm[i, c, :] = X[i, c, :] - mean_val

    # Step 13: Train-test split (80/20)
    print("Applying 80/20 Train-Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Step 14: Save (.npy)
    out_prefix = f"BCICIV_ds3_{subject_id}"
    out_x_train = os.path.join(PROCESSED_DIR, f"{out_prefix}_X_train.npy")
    out_x_test = os.path.join(PROCESSED_DIR, f"{out_prefix}_X_test.npy")
    out_y_train = os.path.join(PROCESSED_DIR, f"{out_prefix}_y_train.npy")
    out_y_test = os.path.join(PROCESSED_DIR, f"{out_prefix}_y_test.npy")
    
    np.save(out_x_train, X_train)
    np.save(out_x_test, X_test)
    np.save(out_y_train, y_train)
    np.save(out_y_test, y_test)
    
    print(f"Saved for {subject_id}")

if __name__ == '__main__':
    # Find all data_set_IVa_*.mat files in the subdirectories
    search_pattern = os.path.join(DATA_DIR, 'data_set_IVa_*_mat', '100Hz', 'data_set_IVa_*.mat')
    mat_files = glob.glob(search_pattern)
    
    if not mat_files:
        print("No files found!")
    else:
        for file_path in mat_files:
            preprocess_ds3_subject(file_path)
