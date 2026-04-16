import scipy.io as sio
import os

file_path = r'C:\Users\Dell\Desktop\EEG\BCI-4-2a\A01T.mat'
try:
    mat = sio.loadmat(file_path)
    print("Keys in mat:", mat.keys())
    for k in mat.keys():
        if not k.startswith('__'):
            print(f"{k}: type={type(mat[k])}, shape={mat[k].shape if hasattr(mat[k], 'shape') else 'none'}")
except NotImplementedError:
    import h5py
    with h5py.File(file_path, 'r') as f:
        print("Keys in h5py mat:", list(f.keys()))
