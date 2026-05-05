import numpy as np
import os

data_path = "processed_data"

for file in os.listdir(data_path):
    if "X_train" in file:
        prefix = file.replace("_X_train.npy", "")

        X = np.load(os.path.join(data_path, file))
        y = np.load(os.path.join(data_path, prefix + "_y_train.npy"))

        print("="*50)
        print(f"Dataset: {prefix}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        print(f"y dtype: {y.dtype}")