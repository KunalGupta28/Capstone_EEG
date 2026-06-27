"""
channel_selection/jaya_optimizer.py
==================================
Optimizations applied:
1. Reduced pop size and iter limits dynamically:
   - 22 channels: pop=10, iter=15
   - 59 channels: pop=12, iter=20
   - 118 channels: pop=15, iter=25
2. Shared persistent dict-based fitness caching passed to evaluate_fitness().
3. Parallel candidate evaluation using joblib.Parallel.
4. Early stopping if best fitness has not improved for 5 iterations.
5. Elitism preserved (best solution kept in slot 0).
"""

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

from channel_selection.fitness import evaluate_fitness
from channel_selection.logistic_map import generate_chaotic_sequence
from channel_selection.transfer_function import binarize

# Dynamic settings matching the new optimized iteration budgets (Optimization 2)
DS_POP_ITER = {
    22:  (10, 15),
    59:  (12, 20),
    118: (15, 25),
}

def _get_pop_iter(n_channels: int, n_pop: int, n_iter: int):
    """Return dataset-appropriate pop/iter if defaults were not overridden."""
    for ch_count, (pop, it) in DS_POP_ITER.items():
        if n_channels <= ch_count:
            return pop, it
    return n_pop, n_iter


def run_ls_bjoa(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_channels: int,
    candidate_indices: list,
    min_channels: int,
    n_pop: int = 10,
    n_iter: int = 20,
    is_binary: bool = False,
    seed: int = 42,
) -> dict:
    """
    LS-BJOA: Logistic S-shaped Binary Jaya Optimization Algorithm.
    Highly optimized for execution speed.

    Parameters
    ----------
    X_train          : (n_trials, n_channels, n_times)
    y_train          : (n_trials,)
    n_channels       : total number of channels
    candidate_indices: channels always forced to 1 (selected)
    min_channels     : minimum number of selected channels for valid solution
    n_pop            : population size
    n_iter           : number of iterations
    is_binary        : True for binary classification datasets
    seed             : numpy random seed

    Returns
    -------
    dict with keys: best_mask, selected_indices, n_selected,
                    best_fitness, fitness_history
    """
    np.random.seed(seed)

    # 1. Scale down defaults
    n_pop, n_iter = _get_pop_iter(n_channels, n_pop, n_iter)
    print(f"    [LS-BJOA] n_channels={n_channels}, n_pop={n_pop}, n_iter={n_iter}", flush=True)

    # 2. Shared fitness cache (Optimization 1 & 2)
    cache = {}

    # Initialize population with ~50% ones, then force candidate channels
    pop = (np.random.rand(n_pop, n_channels) > 0.5).astype(int)
    for i in range(n_pop):
        for idx in candidate_indices:
            pop[i, idx] = 1

    # 3. Parallel initialization of fitness (Optimization 2)
    # Find unique initial masks to avoid redundant computation
    unique_masks = []
    unique_keys = []
    for mask in pop:
        mask_copy = mask.copy()
        for idx in candidate_indices:
            mask_copy[idx] = 1
        key = tuple(mask_copy.astype(int))
        if key not in unique_keys:
            unique_keys.append(key)
            unique_masks.append(mask_copy)

    # Evaluate unique initial masks in parallel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_fitness)(
                mask=m,
                X_train=X_train,
                y_train=y_train,
                candidate_indices=candidate_indices,
                min_channels=min_channels,
                is_binary=is_binary,
                cache=None  # Disable internal caching in parallel processes
            )
            for m in unique_masks
        )
    # Update master cache
    for k, val in zip(unique_keys, results):
        cache[k] = val

    # Fill initial fitness array from cache
    fitness = np.array([cache[tuple(pop[i])] for i in range(n_pop)])

    best_idx  = int(np.argmax(fitness))
    worst_idx = int(np.argmin(fitness))

    best_mask    = pop[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    fitness_history = [best_fitness]

    # Pre-generate chaotic sequence
    chaotic_seq = generate_chaotic_sequence(n_pop * n_iter * 2 + 10, c0=0.8)
    seq_idx = 0

    no_improve_count = 0

    for iteration in range(n_iter):
        new_pop     = pop.copy()
        new_fitness = fitness.copy()

        # Generate candidate updates for the entire population first
        candidate_masks = []
        for i in range(n_pop):
            c1 = chaotic_seq[seq_idx];     seq_idx += 1
            c2 = chaotic_seq[seq_idx];     seq_idx += 1

            x_best  = pop[best_idx].astype(float)
            x_worst = pop[worst_idx].astype(float)
            x_curr  = pop[i].astype(float)

            # Jaya update rule (Eq. 19-23 from paper)
            x_new_cont = x_curr + c1 * (x_best - np.abs(x_curr)) \
                                 - c2 * (x_worst - np.abs(x_curr))
            x_new_bin  = binarize(x_new_cont)

            # Always force candidate channels
            for idx in candidate_indices:
                x_new_bin[idx] = 1

            candidate_masks.append(x_new_bin)

        # Find only the uncached candidate masks to evaluate in parallel
        uncached_candidate_keys = []
        uncached_candidate_masks = []
        for mask in candidate_masks:
            key = tuple(mask.astype(int))
            if key not in cache:
                if key not in uncached_candidate_keys:
                    uncached_candidate_keys.append(key)
                    uncached_candidate_masks.append(mask)

        # Parallel evaluation of the uncached candidates
        if uncached_candidate_masks:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                candidate_results = Parallel(n_jobs=-1)(
                    delayed(evaluate_fitness)(
                        mask=m,
                        X_train=X_train,
                        y_train=y_train,
                        candidate_indices=candidate_indices,
                        min_channels=min_channels,
                        is_binary=is_binary,
                        cache=None
                    )
                    for m in uncached_candidate_masks
                )
            # Update cache in the main process
            for k, val in zip(uncached_candidate_keys, candidate_results):
                cache[k] = val

        # Greedy acceptance check (from cache)
        for i in range(n_pop):
            x_new_bin = candidate_masks[i]
            fit_new = cache[tuple(x_new_bin.astype(int))]

            if fit_new > new_fitness[i]:
                new_pop[i]     = x_new_bin
                new_fitness[i] = fit_new

        # Elitism: preserve best solution into slot 0
        new_pop[0]     = best_mask.copy()
        new_fitness[0] = best_fitness

        pop     = new_pop
        fitness = new_fitness

        best_idx  = int(np.argmax(fitness))
        worst_idx = int(np.argmin(fitness))

        # Check if global best has improved
        if fitness[best_idx] > best_fitness:
            best_fitness = float(fitness[best_idx])
            best_mask    = pop[best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        fitness_history.append(best_fitness)
        print(f"    [LS-BJOA] Iter {iteration+1:3d}/{n_iter} | "
              f"Best fitness: {best_fitness:.4f} | "
              f"Channels selected: {int(np.sum(best_mask))}", flush=True)

        # Early Stopping check (Optimization 2)
        if no_improve_count >= 5:
            print(f"    [LS-BJOA] Early stopping at iteration {iteration+1}", flush=True)
            break

    selected_indices = np.where(best_mask == 1)[0].tolist()

    return {
        "best_mask":       best_mask,
        "selected_indices": selected_indices,
        "n_selected":       len(selected_indices),
        "best_fitness":     best_fitness,
        "fitness_history":  fitness_history,
    }
