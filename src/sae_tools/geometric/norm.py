import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_matrix(W_dec):
    """
    Perform basic statistical analysis on the decoder matrix
    W_dec shape expectation: [d_sae, d_model] -> [num_features, hidden_size]
    """
    print(f"\n{'='*40}")
    print("Matrix Analysis")
    print(f"{'='*40}")
    
    # 1. Shape check
    d_sae, d_model = W_dec.shape
    print(f"Shape: {W_dec.shape}")
    print(f"  - Number of Features (d_sae): {d_sae}")
    print(f"  - Model Dimension (d_model):  {d_model}")
    print(f"  - Dtype: {W_dec.dtype}")

    # 2. Norm Check
    # The decoder vectors are usually normalized to unit length, or close to unit length
    # Calculate the L2 norm for each feature vector
    feature_norms = torch.norm(W_dec, p=2, dim=1) # dim=1 represents the norm along the model dimension
    
    mean_norm = feature_norms.mean().item()
    std_norm = feature_norms.std().item()
    min_norm = feature_norms.min().item()
    max_norm = feature_norms.max().item()

    print("\nFeature Norm Statistics (L2):")
    print(f"  - Mean: {mean_norm:.4f}")
    print(f"  - Std:  {std_norm:.4f}")
    print(f"  - Min:  {min_norm:.4f}")
    print(f"  - Max:  {max_norm:.4f}")

    if 0.9 < mean_norm < 1.1 and std_norm < 0.1:
        print("\n[Analysis] ✅ Most feature vectors seem to be normalized (unit length).")
    else:
        print("\n[Analysis] ⚠️ Feature vectors are NOT strictly unit length. (This depends on training setup)")

    return feature_norms

def visualize_norms(feature_norms, output_path=None):
    """Plot the histogram of the norm distribution"""
    if output_path:
        plt.figure(figsize=(10, 6))
        plt.hist(feature_norms.numpy(), bins=100, log=True)
        plt.title("Distribution of Decoder Feature Norms")
        plt.xlabel("L2 Norm")
        plt.ylabel("Count (Log Scale)")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path)
        print(f"\n>>> Histogram saved to {output_path}")