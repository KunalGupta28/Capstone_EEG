import numpy as np
import os

data_path = "processed_data"

if not os.path.exists(data_path):
    print(f"Directory not found: {data_path}")
else:
    for root, dirs, files in os.walk(data_path):
        for file in sorted(files):
            if file.endswith("_X_train.npy"):
                prefix = file.replace("_X_train.npy", "")
                y_file = prefix + "_y_train.npy"
            elif file.endswith("_X.npy"):
                prefix = file.replace("_X.npy", "")
                y_file = prefix + "_y.npy"
            else:
                continue

            if y_file in files:
                X = np.load(os.path.join(root, file))
                y = np.load(os.path.join(root, y_file))

                print("="*50)
                print(f"Dataset File: {os.path.join(os.path.basename(root), file)}")
                print(f"X shape: {X.shape}")
                print(f"y shape: {y.shape}")
                print(f"Unique labels: {np.unique(y)}")
                print(f"y dtype: {y.dtype}")