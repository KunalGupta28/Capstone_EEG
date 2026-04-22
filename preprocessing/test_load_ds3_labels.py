import scipy.io as sio
import numpy as np

file_path = r'C:\Users\Dell\Desktop\EEG\BCI -III-IVa\data_set_IVa_aa_mat\100Hz\data_set_IVa_aa.mat'
mat = sio.loadmat(file_path)

mrk = mat.get('mrk')
y = mrk['y'][0,0][0]
print("y unique values:", np.unique(y[~np.isnan(y)]))
print("Number of NaNs:", np.sum(np.isnan(y)))
print("Total y length:", len(y))

# Let's check another file 'al' just to be sure
file_path2 = r'C:\Users\Dell\Desktop\EEG\BCI -III-IVa\data_set_IVa_al_mat\100Hz\data_set_IVa_al.mat'
mat2 = sio.loadmat(file_path2)
y2 = mat2.get('mrk')['y'][0,0][0]
print("\ny2 unique values:", np.unique(y2[~np.isnan(y2)]))
print("Number of NaNs in y2:", np.sum(np.isnan(y2)))
print("Total y2 length:", len(y2))
