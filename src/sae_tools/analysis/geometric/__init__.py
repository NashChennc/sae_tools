"""Geometric analysis utilities.

Import concrete functions from submodules to avoid importing optional RAPIDS
dependencies unless UMAP is actually used.
"""

__all__ = [
    "norm",
    "similarity",
    "umap",
]
