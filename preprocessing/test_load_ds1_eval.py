import scipy.io as sio

file_path = r'C:\Users\Dell\Desktop\EEG\BCICIV_1_mat\BCICIV_eval_ds1a.mat'
mat = sio.loadmat(file_path)

cnt = mat['cnt']
if 'mrk' in mat:
    mrk = mat['mrk']
    print("mrk present in eval!")
    print(mrk.dtype.names)
else:
    print("mrk not present in eval. We might need true labels from somewhere else.")
    
print("Keys:", mat.keys())
