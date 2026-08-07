import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_channel_selection_frequency(
    selected_indices_list: list,
    n_channels: int,
    dataset_name: str,
    save_path: str,
) -> None:
    """
    Bar chart of how often each channel was selected across subjects.
    Save as PNG. Do NOT use mne.viz.plot_topomap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    counts = np.zeros(n_channels)
    for indices in selected_indices_list:
        counts[indices] += 1
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(n_channels), counts, color='teal')
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Selection Frequency')
    ax.set_title(f'Channel Selection Frequency - {dataset_name}')
    ax.set_xticks(range(n_channels))
    
    # Only label every few ticks if there are many channels to avoid crowding
    if n_channels > 30:
        ax.set_xticks(range(0, n_channels, 5))
        
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_fitness_history(
    fitness_history: list,
    subject_id: str,
    dataset_name: str,
    save_path: str,
) -> None:
    """Line plot of best fitness per iteration. Save as PNG."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(fitness_history)), fitness_history, marker='o', linestyle='-', color='purple')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Fitness')
    ax.set_title(f'LS-BJOA Fitness History - {dataset_name} ({subject_id})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
