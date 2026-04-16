import numpy as np
import os

x_path = r'C:\Users\Dell\Desktop\EEG\processed_data\A01T_X.npy'
y_path = r'C:\Users\Dell\Desktop\EEG\processed_data\A01T_y.npy'

X = np.load(x_path)
y = np.load(y_path)

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)

# Check normalization
print(f"X mean overall: {np.mean(X):.4f}")
print(f"X std overall: {np.std(X):.4f}")
print(f"Label distribution: {np.unique(y, return_counts=True)}")
