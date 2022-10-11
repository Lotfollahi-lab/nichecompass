from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set


def _compute_graph_indices_and_distances(adata: AnnData,
                                         feature_key: str,
                                         n_neighbors: int,
                                         mode: Literal["knn", "umap"]="knn"):
    """
    Compute a k-nearest-neighbors (knn) graph based on feature values stored in
    an AnnData object and return the indices of and distances to the k-nearest 
    neighbors for each observation.

    Parameters
    ----------
    adata:
        AnnData object with feature values for distance calculation stored in 
        ´adata.obsm[feature_key]´.
    feature_key:
        Key under which the feature values for distance calculation are stored 
        in ´adata.obsm´.
    n_neighbors:
        Number of neighbors for the knn graph.
    mode:

    Returns
    ----------
    knn_indices:
        NumPy array that contains the indices of the ´n_neighbors´ nearest 
        neighbors for each observation.
    knn_distances:
        NumPy array that contains the distances to the ´n_neighbors´ nearest 
        neighbors for each observation.
    """
    X = adata.obsm[feature_key]
    if mode == "knn":
    distances = pairwise_distances(X=X, metric="euclidean")

    # Retrieve knn indices and knn distances from pairwise distances
    obs_range = np.arange(distances.shape[0])[:, None]
    knn_indices = np.argpartition(distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
    knn_indices = knn_indices[obs_range, np.argsort(distances[obs_range,
                                                    knn_indices])]
    knn_distances = distances[obs_range, knn_indices]

    elif mode == "umap":
    knn_indices, knn_dists, _ = nearest_neighbors(
        X,
        n_neighbors,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        angular=angular,
        verbose=verbose,
    )
    return knn_indices, knn_distances


def _compute_graph_connectivities(
        adata: AnnData,
        feature_key: str,
        n_neighbors: int,
        mode: Literal["knn", "umap"]="knn",
        seed: int=42) -> coo_matrix:
    """
    Compute graph connectivities from a feature stored in an AnnData object. The
    graph can be a simple k-nearest-neighbors graph (´mode´ == "knn") or a 
    fuzzy simplical set (´mode´ == "umap") as in McInnes, L., Healy, J. & 
    Melville, J. UMAP: Uniform Manifold Approximation and Projection for 
    Dimension Reduction. arXiv [stat.ML] (2018).

    Parameters
    ----------
    adata:
    feature_key:
    n_neighbors:
    mode:
    seed:

    Returns
    ----------
    connectivities:

    """ 
    # Get features for graph construction
    X = adata.obsm[feature_key]

    # Compute graph connectivities
    if mode == "knn":
        connectivities = kneighbors_graph(X=X, n_neighbors=n_neighbors)
    elif mode == "umap":
        knn_indices, knn_distances = _compute_knn_graph_indices_and_distances(
            adata=adata,
            feature_key=feature_key,
            n_neighbors=n_neighbors)

        connectivities = fuzzy_simplicial_set(
            X=X,
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


def _convert_to_one_hot(vector: np.ndarray,
                        n_classes: Optional[int]):
    """
    Converts an input 1D vector of integer labels into a 2D array of one-hot 
    vectors, where for an i'th input value of j, a '1' will be inserted in the 
    i'th row and j'th column of the output one-hot vector. Adapted from 
    https://github.com/theislab/scib/blob/29f79d0135f33426481f9ff05dd1ae55c8787142/scib/metrics/lisi.py#L498.

    Parameters
    ----------
    vector:
        Vector to be one-hot-encoded.
    n_classes:
        Number of classes to be considered for one-hot-encoding. If ´None´, the
        number of classes will be inferred from ´vector´.

    Returns
    ----------
    one_hot:
        2D NumPy array of one-hot-encoded vectors.

    Example:
    ´´´
    vector = np.array((1, 0, 4))
    one_hot = _convert_to_one_hot(vector)
    print(one_hot)
    [[0 1 0 0 0]
     [1 0 0 0 0]
     [0 0 0 0 1]]
    ´´´
    """
    if n_classes is None:
        n_classes = np.max(vector) + 1
    one_hot = np.zeros(shape=(len(vector), n_classes))
    one_hot[np.arange(len(vector)), vector] = 1
    return one_hot.astype(int)
