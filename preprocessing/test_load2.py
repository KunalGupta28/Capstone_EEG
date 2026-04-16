import scipy.io as sio
import numpy as np

file_path = r'C:\Users\Dell\Desktop\EEG\BCI-4-2a\A01T.mat'
mat = sio.loadmat(file_path)
data = mat['data']

print("Data shape:", data.shape)
for i in range(data.shape[1]):
    run_data = data[0, i]
    print(f"\nRun {i}:")
    print("Type:", type(run_data))
    if isinstance(run_data, np.ndarray) and run_data.dtype.names:
        print("Fields:", run_data.dtype.names)
        for field in run_data.dtype.names:
            val = run_data[field]
            print(f"  {field}: shape={val.shape if hasattr(val, 'shape') else 'none'}")
    elif isinstance(run_data, np.ndarray):
        print(f"  Shape: {run_data.shape}")
        
