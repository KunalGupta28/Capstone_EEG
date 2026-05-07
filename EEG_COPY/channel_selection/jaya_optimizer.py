import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from channel_selection.fitness import evaluate_fitness
from channel_selection.logistic_map import generate_chaotic_sequence
from channel_selection.transfer_function import binarize

# Dataset-specific search settings (Issue 9)
DS_POP_ITER = {
    22:  (10, 20),   # BCI-4-2a
    59:  (20, 30),   # BCI_IV_1
    118: (30, 40),   # BCI_III_IVa
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

    Parameters
    ----------
    X_train          : (n_trials, n_channels, n_times)
    y_train          : (n_trials,)
    n_channels       : total number of channels
    candidate_indices: channels always forced to 1 (selected)
    min_channels     : minimum number of selected channels for valid solution
    n_pop            : population size (auto-scaled by n_channels if default)
    n_iter           : number of iterations (auto-scaled if default)
    is_binary        : True for binary classification datasets
    seed             : numpy random seed

    Returns
    -------
    dict with keys: best_mask, selected_indices, n_selected,
                    best_fitness, fitness_history
    """
    np.random.seed(seed)

    # Auto-scale population and iterations to search space size (Issue 9)
    n_pop, n_iter = _get_pop_iter(n_channels, n_pop, n_iter)
    print(f"    [LS-BJOA] n_channels={n_channels}, n_pop={n_pop}, n_iter={n_iter}", flush=True)

    # Initialize population with ~50% ones, then force candidate channels
    pop = (np.random.rand(n_pop, n_channels) > 0.5).astype(int)
    for i in range(n_pop):
        for idx in candidate_indices:
            pop[i, idx] = 1

    # Suppress only SVM ConvergenceWarning (Issue 11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitness = np.array([
            evaluate_fitness(pop[i], X_train, y_train, candidate_indices,
                             min_channels, is_binary)
            for i in range(n_pop)
        ])

    best_idx  = int(np.argmax(fitness))
    worst_idx = int(np.argmin(fitness))

    best_mask    = pop[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    fitness_history = [best_fitness]

    # Chaotic sequence long enough for all moves
    chaotic_seq = generate_chaotic_sequence(n_pop * n_iter * 2 + 10, c0=0.8)
    seq_idx = 0

    for iteration in range(n_iter):
        new_pop     = pop.copy()
        new_fitness = fitness.copy()

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

            # Always force candidate channels (Issue 5)
            for idx in candidate_indices:
                x_new_bin[idx] = 1

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fit_new = evaluate_fitness(x_new_bin, X_train, y_train,
                                          candidate_indices, min_channels, is_binary)

            # Greedy acceptance
            if fit_new > new_fitness[i]:
                new_pop[i]     = x_new_bin
                new_fitness[i] = fit_new

        # Issue 10: Elitism — preserve best solution into slot 0
        new_pop[0]     = best_mask.copy()
        new_fitness[0] = best_fitness

        pop     = new_pop
        fitness = new_fitness

        best_idx  = int(np.argmax(fitness))
        worst_idx = int(np.argmin(fitness))

        if fitness[best_idx] > best_fitness:
            best_fitness = float(fitness[best_idx])
            best_mask    = pop[best_idx].copy()

        fitness_history.append(best_fitness)
        print(f"    [LS-BJOA] Iter {iteration+1:3d}/{n_iter} | "
              f"Best fitness: {best_fitness:.4f} | "
              f"Channels selected: {int(np.sum(best_mask))}", flush=True)

    selected_indices = np.where(best_mask == 1)[0].tolist()

    return {
        "best_mask":       best_mask,
        "selected_indices": selected_indices,
        "n_selected":       len(selected_indices),
        "best_fitness":     best_fitness,
        "fitness_history":  fitness_history,
    }
