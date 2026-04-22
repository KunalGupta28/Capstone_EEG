import scipy.io as sio

file_path = r'C:\Users\Dell\Desktop\EEG\BCI -III-IVa\data_set_IVa_aa_mat\100Hz\data_set_IVa_aa.mat'
mat = sio.loadmat(file_path)
print("Keys:", mat.keys())

for k in mat.keys():
    if not k.startswith('__'):
        print(f"Key: {k}, Type: {type(mat[k])}")
        if hasattr(mat[k], 'shape'):
            print(f"  Shape: {mat[k].shape}")
            
cnt = mat.get('cnt')
mrk = mat.get('mrk')
nfo = mat.get('nfo')

if mrk is not None:
    print("\n--- mrk ---")
    print("mrk dtype:", mrk.dtype)
    if mrk.dtype.names:
        for field in mrk.dtype.names:
            print(f"  {field}: shape {mrk[field][0,0].shape}")

if nfo is not None:
    print("\n--- nfo ---")
    print("nfo dtype:", nfo.dtype)
    if nfo.dtype.names:
        for field in nfo.dtype.names:
            val = nfo[field][0,0]
            print(f"  {field}: shape {val.shape if hasattr(val, 'shape') else 'none'}, type: {type(val)}")
            if field == 'fs':
                print(f"    value: {val}")
            elif field == 'classes':
                print(f"    value: {val}")
