import numpy as np
from channel_selection.fitness import evaluate_fitness
from channel_selection.logistic_map import generate_chaotic_sequence
from channel_selection.transfer_function import binarize

def run_ls_bjoa(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_channels: int,
    candidate_indices: list,
    min_channels: int,
    n_pop: int = 10,
    n_iter: int = 20,
    w1: float = 0.9,
    w2: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Returns:
      {
        'best_mask': np.ndarray (binary, length=n_channels),
        'selected_indices': list of int,
        'n_selected': int,
        'best_fitness': float,
        'fitness_history': list of float (best fitness per iteration),
      }
    """
    np.random.seed(seed)
    
    # Initialize population randomly
    pop = np.random.randint(0, 2, size=(n_pop, n_channels))
    for i in range(n_pop):
        for idx in candidate_indices:
            pop[i, idx] = 1
            
    # Evaluate initial population
    fitness = np.zeros(n_pop)
    for i in range(n_pop):
        fitness[i] = evaluate_fitness(pop[i], X_train, y_train, candidate_indices, min_channels, w1, w2)
        
    best_idx = np.argmax(fitness)
    worst_idx = np.argmin(fitness)
    
    best_mask = pop[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    fitness_history = [float(best_fitness)]
    
    # Generate chaotic sequence. Need 2 values per generation per pop
    chaotic_seq = generate_chaotic_sequence(n_pop * n_iter * 2, c0=0.8)
    seq_idx = 0
    
    for iteration in range(n_iter):
        new_pop = np.zeros_like(pop)
        new_fitness = np.zeros(n_pop)
        
        for i in range(n_pop):
            c1 = chaotic_seq[seq_idx]
            c2 = chaotic_seq[seq_idx+1]
            seq_idx += 2
            
            x_best = pop[best_idx]
            x_worst = pop[worst_idx]
            x_curr = pop[i]
            
            x_new_cont = x_curr + c1 * (x_best - np.abs(x_curr)) - c2 * (x_worst - np.abs(x_curr))
            x_new_bin = binarize(x_new_cont)
            
            for idx in candidate_indices:
                x_new_bin[idx] = 1
                
            fit_new = evaluate_fitness(x_new_bin, X_train, y_train, candidate_indices, min_channels, w1, w2)
            
            if fit_new > fitness[i]:
                new_pop[i] = x_new_bin
                new_fitness[i] = fit_new
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]
                
        pop = new_pop
        fitness = new_fitness
        
        best_idx = np.argmax(fitness)
        worst_idx = np.argmin(fitness)
        
        if fitness[best_idx] > best_fitness:
            best_fitness = fitness[best_idx]
            best_mask = pop[best_idx].copy()
            
        fitness_history.append(float(best_fitness))
        
    selected_indices = np.where(best_mask == 1)[0].tolist()
    
    return {
        'best_mask': best_mask,
        'selected_indices': selected_indices,
        'n_selected': len(selected_indices),
        'best_fitness': float(best_fitness),
        'fitness_history': fitness_history,
    }
