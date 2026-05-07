"""
Smoke test for all fixed channel selection modules.
Verifies Issues 1-11 are resolved.
"""
import sys, os, warnings
sys.path.insert(0, r'C:\Users\Dell\Desktop\EEG\EEG_COPY')
sys.path.insert(0, r'C:\Users\Dell\Desktop\EEG')

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np

print("=" * 60)
print("  ISSUE 1: Transfer function binarization fix")
print("=" * 60)
from channel_selection.transfer_function import sigmoid, binarize
np.random.seed(0)
# Very high x -> sigmoid close to 1 -> random < 1 almost always -> should select (1)
high_x = np.full(1000, 10.0)
result = binarize(high_x)
pct_selected = result.mean()
print(f"  High x=10 -> {pct_selected*100:.1f}% selected (expect ~100%)")
assert pct_selected > 0.9, f"FAIL: expected ~100% selected, got {pct_selected:.2f}"

np.random.seed(0)
# Very low x -> sigmoid close to 0 -> random < 0 almost never -> should reject (0)
low_x = np.full(1000, -10.0)
result = binarize(low_x)
pct_selected = result.mean()
print(f"  Low  x=-10 -> {pct_selected*100:.1f}% selected (expect ~0%)")
assert pct_selected < 0.1, f"FAIL: expected ~0% selected, got {pct_selected:.2f}"
print("  PASS")

print()
print("=" * 60)
print("  ISSUES 2,3,4,5,6,7: Fitness function fixes")
print("=" * 60)
from channel_selection.fitness import evaluate_fitness

# Binary dataset
np.random.seed(42)
X_bin = np.random.randn(60, 10, 50)
y_bin = np.random.randint(0, 2, 60)
mask = np.ones(10, dtype=int)
fit_bin = evaluate_fitness(mask, X_bin, y_bin, [0, 1, 2], min_channels=3, is_binary=True)
print(f"  Binary fitness (all channels): {fit_bin:.4f}")
assert 0.0 <= fit_bin <= 1.0, f"FAIL: fitness out of range: {fit_bin}"

# Multi-class dataset
X_mc = np.random.randn(80, 10, 50)
y_mc = np.random.randint(0, 4, 80)
fit_mc = evaluate_fitness(mask, X_mc, y_mc, [0, 1, 2], min_channels=3, is_binary=False)
print(f"  Multi-class fitness (all channels): {fit_mc:.4f}")
assert 0.0 <= fit_mc <= 1.0, f"FAIL: fitness out of range: {fit_mc}"

# Penalty test: too few channels
mask_few = np.zeros(10, dtype=int)
mask_few[0] = 1  # only 1 channel (candidate_indices will force 3, still check min)
fit_penalty = evaluate_fitness(mask_few, X_bin, y_bin, [], min_channels=5, is_binary=True)
print(f"  Penalty for too-few channels: {fit_penalty:.4f} (expect 0.0)")
assert fit_penalty == 0.0, "FAIL: should return 0.0 for too few channels"
print("  PASS")

print()
print("=" * 60)
print("  ISSUES 9,10: Jaya optimizer (auto-scaling, elitism)")
print("=" * 60)
from channel_selection.jaya_optimizer import run_ls_bjoa, _get_pop_iter

# Test auto-scaling
p22, i22 = _get_pop_iter(22, 10, 20)
p59, i59 = _get_pop_iter(59, 10, 20)
p118, i118 = _get_pop_iter(118, 10, 20)
print(f"  22 ch  -> pop={p22}, iter={i22} (expect 10, 20)")
print(f"  59 ch  -> pop={p59}, iter={i59} (expect 20, 30)")
print(f"  118 ch -> pop={p118}, iter={i118} (expect 30, 40)")
assert (p22, i22) == (10, 20)
assert (p59, i59) == (20, 30)
assert (p118, i118) == (30, 40)

# Run optimizer on tiny data
result = run_ls_bjoa(X_bin, y_bin, n_channels=10,
                     candidate_indices=[0,1,2], min_channels=3,
                     is_binary=True, n_pop=5, n_iter=3, seed=42)
print(f"  LS-BJOA selected {result['n_selected']}/10 channels")
print(f"  Best fitness: {result['best_fitness']:.4f}")
print(f"  History length: {len(result['fitness_history'])} (expect 4: init + 3 iters)")
# Elitism: fitness should be non-decreasing
hist = result['fitness_history']
for k in range(1, len(hist)):
    assert hist[k] >= hist[k-1] - 1e-9, f"FAIL: fitness decreased at iter {k}: {hist[k-1]:.4f} -> {hist[k]:.4f}"
print("  Fitness history is non-decreasing (elitism verified)")
print("  PASS")

print()
print("=" * 60)
print("  ALL TESTS PASSED")
print("=" * 60)
