"""Quick smoke test to verify all channel selection modules import and work."""
import sys, os
sys.path.insert(0, r'C:\Users\Dell\Desktop\EEG\EEG_COPY')
sys.path.insert(0, r'C:\Users\Dell\Desktop\EEG')

import numpy as np

# Test logistic map
from channel_selection.logistic_map import generate_chaotic_sequence
seq = generate_chaotic_sequence(5)
print(f"Chaotic sequence: {seq}")
assert len(seq) == 5
assert all(0 < v < 1 for v in seq[1:])

# Test transfer function
from channel_selection.transfer_function import sigmoid, binarize
s = sigmoid(np.array([0.0, 1.0, -1.0]))
print(f"Sigmoid([0, 1, -1]): {s}")
assert abs(s[0] - 0.5) < 1e-6

b = binarize(np.array([0.0, 5.0, -5.0]))
print(f"Binarize([0, 5, -5]): {b}")
assert b.shape == (3,)

# Test channel_mask
from channel_selection.channel_mask import apply_mask, get_candidate_indices, get_min_channels
print(f"Candidate indices for BCI-4-2a:    {get_candidate_indices('BCI-4-2a')}")
print(f"Candidate indices for BCI_IV_1:    {get_candidate_indices('BCI_IV_1')}")
print(f"Candidate indices for BCI_III_IVa: {get_candidate_indices('BCI_III_IVa')}")
print(f"Min channels for BCI-4-2a:    {get_min_channels('BCI-4-2a')}")
print(f"Min channels for BCI_IV_1:    {get_min_channels('BCI_IV_1')}")
print(f"Min channels for BCI_III_IVa: {get_min_channels('BCI_III_IVa')}")

X_dummy = np.random.randn(10, 22, 100)
X_masked = apply_mask(X_dummy, [0, 5, 10])
print(f"apply_mask: {X_dummy.shape} -> {X_masked.shape}")
assert X_masked.shape == (10, 3, 100)

# Test fitness (quick, with tiny data)
from channel_selection.fitness import evaluate_fitness
X_tiny = np.random.randn(40, 10, 50)
y_tiny = np.random.randint(0, 2, 40)
mask = np.ones(10, dtype=int)
fit = evaluate_fitness(mask, X_tiny, y_tiny, [0, 1, 2], min_channels=3)
print(f"Fitness (all channels): {fit:.4f}")

# Test jaya optimizer (tiny run)
from channel_selection.jaya_optimizer import run_ls_bjoa
result = run_ls_bjoa(X_tiny, y_tiny, n_channels=10, candidate_indices=[0,1,2],
                     min_channels=3, n_pop=5, n_iter=3, seed=42)
print(f"LS-BJOA result: {result['n_selected']}/{10} channels, fitness={result['best_fitness']:.4f}")
print(f"Selected indices: {result['selected_indices']}")
print(f"Fitness history: {result['fitness_history']}")

# Test visualization imports
from visualization.topomap import plot_channel_selection_frequency, plot_fitness_history
print("Visualization imports OK")

print("\n=== ALL SMOKE TESTS PASSED ===")
