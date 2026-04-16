import scipy.io as sio
import numpy as np

file_path = r'C:\Users\Dell\Desktop\EEG\BCI-4-2a\A01T.mat'
mat = sio.loadmat(file_path)
data = mat['data']

run_data = data[0, 3] # typically run 3 is motor imagery task
X = run_data['X'][0, 0]
y = run_data['y'][0, 0]
trial = run_data['trial'][0, 0]
fs = run_data['fs'][0, 0]
classes = run_data['classes'][0, 0]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("trial shape:", trial.shape)
print("fs:", fs)
print("classes:", classes)
