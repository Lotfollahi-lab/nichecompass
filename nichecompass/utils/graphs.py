"""
This module contains utilities to compute nearest neighbor graphs and graph
indices and distances for use by the NicheCompass model.
"""

from typing import Literal, Tuple

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set


def compute_graph_indices_and_distances(
        adata: AnnData,
        feature_key: str,
        n_neighbors: int,
        mode: Literal["knn", "umap"]="knn",
        seed: int=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute indices of and distances to the ´n_neighbors´ nearest neighbors for
    each observation by constructing a nearest neighbors graph based on feature
    values stored in ´adata.obsm[feature_key]´. If ´mode´ == ´knn´, this will be
    an exact knn graph and if ´mode´ == ´umap´, this may be exact but is likely
    approximated via nearest neighbor descent for better performance.

    Parameters
    ----------
    adata:
        AnnData object with feature values for distance calculation stored in
        ´adata.obsm[feature_key]´.
    feature_key:
        Key under which the feature values for distance calculation are stored
        in ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the nearest neighbors graph construction.
    mode:
        If ´knn´, compute an exact k-nearest-neighbors (knn) graph using
        sklearn. If ´umap´, compute a nearest neighbor graph with nearest
        neighbor descent using umap.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    knn_indices:
        2-D NumPy array that contains the indices of the ´n_neighbors´ nearest
        neighbors for each observation.
    knn_distances:
        2-D NumPy array that contains the distances to the ´n_neighbors´ nearest
        neighbors for each observation.
    """
    if mode == "knn":
        # Calculate pairwise feature distances and retrieve knn indices and
        # distances
        distances = pairwise_distances(X=adata.obsm[feature_key],
                                       metric="euclidean")
        obs_range = np.arange(distances.shape[0])[:, None]
        knn_indices = (np.argpartition(distances, n_neighbors - 1, axis=1)
                       [:, :n_neighbors])
        knn_indices = knn_indices[obs_range, np.argsort(distances[obs_range,
                                                        knn_indices])]
        knn_distances = distances[obs_range, knn_indices]
    elif mode == "umap":
        # Calculate knn indices and distances using nearest neighbors descent
        knn_indices, knn_distances, _ = nearest_neighbors(
            X=adata.obsm[feature_key],
            n_neighbors=n_neighbors,
            random_state=seed,
            metric="euclidean",
            metric_kwds=None,
            angular=None)
    return knn_indices, knn_distances


def compute_graph_connectivities(
        adata: AnnData,
        feature_key: str,
        n_neighbors: int,
        mode: Literal["knn", "umap"]="knn",
        seed: int=0) -> sp.csr_matrix:
    """
    Compute graph connectivities by constructing a nearest neighbor graph based
    on ´adata.obsm[feature_key]´.

    Parameters
    ----------
    adata:
        AnnData object with feature values for nearest neighbor computation
        stored in ´adata.obsm[feature_key]´.
    feature_key:
        Key under which the feature values for nearest neighbor computation are
        stored in ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the nearest neighbors graph construction.
    mode:
        If ´knn´, computes exact connectivities from a knn graph using sklearn.
        If ´umap´, computes a fuzzy simplical set based on an approximated
        neighbor graph.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    connectivities:
        Sparse matrix that contains the nearest neighbor graph connectivity 
        between all observations.
    """
    # Compute exact graph connectivities
    if mode == "knn":
        connectivities = kneighbors_graph(X=adata.obsm[feature_key],
                                          n_neighbors=n_neighbors)
    # Compute graph connectivities using a fuzzy simplical set and approximate
    # neighbor graph
    elif mode == "umap":
        knn_indices, knn_distances = compute_graph_indices_and_distances(
            adata=adata,
            feature_key=feature_key,
            n_neighbors=n_neighbors,
            mode="umap",
            seed=seed)
        connectivities = fuzzy_simplicial_set(
            X=adata.obsm[feature_key],
            n_neighbors=n_neighbors,
            random_state=seed,
            metric="euclidean",
            knn_indices=knn_indices,
            knn_dists=knn_distances)
        if isinstance(connectivities, tuple):
            # In umap-learn 0.4, fuzzy_simplical_set() returns
            # (result, sigmas, rhos)
            connectivities = connectivities[0]
    return connectivities
