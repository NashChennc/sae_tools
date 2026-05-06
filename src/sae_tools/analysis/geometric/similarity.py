import seaborn as sns
import matplotlib.colors as mcolors
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_knn_subset(data_norm, seed_indices, k=1000):
    """
    Input:
        data_norm: normalized full data (N, D) on GPU
        seed_indices: list of indices of the core points you are interested in
        k: number of neighbors to retain for each seed
    Output:
        subset_data: filtered data (M, D) on CPU (ready to feed to UMAP)
        subset_indices: filtered original indices (M,)
        mask_seeds: a boolean mask, used to mark which ones are the original seeds in the subset
    """
    # 1. Get the seed vectors (S, D)
    seeds = data_norm[seed_indices]
    
    # 2. Calculate the similarity matrix (N, S)
    # Cosine Similarity = A . B^T (because it is normalized)
    sim_matrix = torch.mm(data_norm, seeds.t())
    
    # 3. For each seed, take the Top-K
    # values: (k, S), indices: (k, S)
    _, topk_indices = torch.topk(sim_matrix, k=k, dim=0)
    
    # 4. Flatten and remove duplicates (Union)
    # This step merges all related "circles"
    all_indices = torch.unique(topk_indices.flatten())
    
    # 5. Prepare the output
    subset_indices = all_indices.cpu().numpy()
    subset_data = data_norm[all_indices].cpu().numpy()
    
    # Mark which points are the original seeds (used for highlighting in the plot)
    # Use np.isin to determine
    mask_seeds = np.isin(subset_indices, seed_indices)
    
    print(f"Original data: {data_norm.shape[0]}, filtered: {len(subset_indices)} (compression rate: {len(subset_indices)/data_norm.shape[0]:.1%})")
    return subset_data, subset_indices, mask_seeds

def plot_crosscorrelation_heatmap(data_norm, seed_indices):
    """
    Plot two heatmaps:
    1. Similarity between seeds (check for redundancy)
    2. Similarity between seeds and the top k neighbors (check for local density)
    """
    seeds = data_norm[seed_indices]
    
    # --- Figure 1: Similarity between seeds ---
    # Calculate the cross-correlation matrix (S x S)
    sim_matrix_seeds = torch.mm(seeds, seeds.t()).cpu().numpy()
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(sim_matrix_seeds, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=seed_indices, yticklabels=seed_indices,
                vmin=-1, vmax=1)
    plt.title("Cosine Similarity Between Seeds (Redundancy Check)")
    plt.show()

def plot_topk_similarity_heatmap(data_norm, seed_indices, k=50):
    seeds = data_norm[seed_indices]

    # --- Figure 2: Similarity between seeds and the top k neighbors ---
    # Calculate the similarity between each seed and the full data (S x N) -> take the TopK
    all_sims = torch.mm(seeds, data_norm.t())
    topk_vals, _ = torch.topk(all_sims, k=k, dim=1)
    topk_vals = topk_vals.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    # Use viridis color: yellow=high similarity, purple=low similarity
    sns.heatmap(topk_vals, cmap="viridis", 
                xticklabels=10, # Every 10 ticks
                yticklabels=seed_indices,
                vmin=0, vmax=1)
    plt.title(f"Top-{k} Neighbors Similarity Heatmap (Sharpness Check)")
    plt.xlabel("Neighbor Rank")
    plt.ylabel("Seed Index")
    plt.show()