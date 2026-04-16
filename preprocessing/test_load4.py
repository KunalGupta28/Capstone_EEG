import scipy.io as sio
import numpy as np

file_path = r'C:\Users\Dell\Desktop\EEG\BCI-4-2a\A01T.mat'
mat = sio.loadmat(file_path)
data = mat['data']

for i in range(data.shape[1]):
    run_data = data[0, i]
    X = run_data['X'][0, 0]
    y = run_data['y'][0, 0]
    trial = run_data['trial'][0, 0]
    print(f"Run {i}: X shape={X.shape}, y shape={y.shape}, trial shape={trial.shape}")
