import seaborn as sns
import matplotlib.colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

try:
    import cudf
    from cuml.manifold import UMAP
except ImportError:
    cudf = None
    UMAP = None

def get_density_fast(x, y, bins=100, sigma=2):
    """
    Quickly calculate the approximate value of point density.
    :param bins: grid resolution, higher is finer but slower
    :param sigma: Gaussian blur radius, controls the smoothness
    """
    # 1. calculate 2D histogram (O(N))
    hh, locx, locy = np.histogram2d(x, y, bins=bins)
    hh_smooth = gaussian_filter(hh, sigma=sigma)
    
    w_x = np.digitize(x, locx) - 1
    w_y = np.digitize(y, locy) - 1
    
    w_x = np.clip(w_x, 0, bins - 1)
    w_y = np.clip(w_y, 0, bins - 1)
    
    z = hh_smooth[w_x, w_y]
    return z

def visualize_density(x, y, z, selected_indices):
    paper_bg = '#FAF9F6'  # Off-white / Eggshell
    paper_bg = '#FFFFFF'  # Pure white
    plt.rcParams['figure.facecolor'] = paper_bg
    plt.rcParams['axes.facecolor'] = paper_bg
    plt.rcParams['font.family'] = 'serif'


    fig, ax = plt.subplots(figsize=(10, 8))

    highlight = '#8E44AD'

    cmap = sns.cubehelix_palette(start=2.8, rot=0.1, gamma=1.0, hue=1, light=0.8, dark=0.4, as_cmap=True)
    cmap = sns.color_palette("crest", as_cmap=True)
    colors = ["#D0D0D0", "#A0A0A0", "#707070"]
    cmap = mcolors.LinearSegmentedColormap.from_list("stone", colors)

    highlight = '#2980B9'
    highlight = '#D9534F'


    scatter = ax.scatter(
        x, y, 
        c=z, 
        s=1, 
        cmap=cmap,
        alpha=1.0
    )

    ax.scatter(
        x[selected_indices], y[selected_indices], 
        c=highlight, 
        s=40, 
        edgecolors=paper_bg, # border color is consistent with background, forming a hollow effect
        linewidth=1.5,
        zorder=10,
        label='Selected'
    )

    ax.axis('off')

    plt.colorbar(scatter, label='Local Density')
    plt.title(f'UMAP Projection of 60k Points (Colored by Density)')
    plt.show()

def run_umap(data):
    if cudf is None or UMAP is None:
        raise ImportError(
            "RAPIDS dependencies are required for run_umap. "
            "Install cudf and cuml before running geometric UMAP analysis."
        )

    try:
        gpu_data = cudf.DataFrame(data)
    except RuntimeError as exc:
        raise RuntimeError(
            "cuDF requires a working CUDA context. "
            "Ensure an NVIDIA GPU with CUDA drivers is available."
        ) from exc

    umap_gpu = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.01,
        spread=5.0,
        metric='cosine',
        init='spectral',
        repulsion_strength=2.0,
        random_state=19260817
    )

    embedding = umap_gpu.fit_transform(gpu_data)

    # 3. convert back to Numpy for plotting
    embedding_numpy = embedding.to_pandas().values

    x = embedding_numpy[:, 0]
    y = embedding_numpy[:, 1]
    z = get_density_fast(x, y, bins=200, sigma=4)

    return x, y, z, embedding_numpy
