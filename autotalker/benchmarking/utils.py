from typing import Tuple, Literal, Optional

import numpy as np
from anndata import AnnData
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set


def _compute_graph_indices_and_distances(
        adata: AnnData,
        feature_key: str,
        n_neighbors: int,
        mode: Literal["knn", "umap"]="knn",
        seed: int=42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute indices of and distances to the  ´n_neighbors´ nearest neighbors
    for each observation by constructing a nearest neighbors graph based on 
    feature values stored in the  ´obsm´ attribute of an AnnData object.
    If ´mode´ == "knn", this will be an exact knn graph and if ´mode´ == "umap",
    this may be exact but is likely approximated via nearest neighbor descent
    for better performance.

    Parameters
    ----------
    adata:
        AnnData object with feature values for distance calculation stored in 
        ´adata.obsm[feature_key]´.
    feature_key:
        Key under which the feature values for distance calculation are stored 
        in ´adata.obsm´.
    n_neighbors:
        Number of neighbors for the nearest neighbors graph construction.
    mode:
        If "knn", compute an exact k-nearest-neighbors (knn) graph using sklearn.
        If "umap", compute a nearest neighbor graph with nearest neighbor 
        descent using umap.
    seed:
        Random seed to get reproducible results.

    Returns
    ----------
    knn_indices:
        2-D NumPy array that contains the indices of the ´n_neighbors´ nearest 
        neighbors for each observation.
    knn_distances:
        2-D NumPy array that contains the distances to the ´n_neighbors´ nearest 
        neighbors for each observation.
    """
    # Get features for graph construction
    X = adata.obsm[feature_key]

    if mode == "knn":
        # Calculate pairwise feature distances and retrieve knn indices and 
        # distances
        distances = pairwise_distances(X=X, metric="euclidean")
        obs_range = np.arange(distances.shape[0])[:, None]
        knn_indices = np.argpartition(distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
        knn_indices = knn_indices[obs_range, np.argsort(distances[obs_range,
                                                        knn_indices])]
        knn_distances = distances[obs_range, knn_indices]
    elif mode == "umap":
        # Calculate knn indices and distances using nearest neighbors descent
        knn_indices, knn_distances, _ = nearest_neighbors(
            X=X,
            n_neighbors=n_neighbors,
            random_state=seed,
            metric="euclidean",
            metric_kwds=None,
            angular=None)

    return knn_indices, knn_distances


def _compute_graph_connectivities(
        adata: AnnData,
        feature_key: str,
        n_neighbors: int,
        mode: Literal["knn", "umap"]="knn",
        seed: int=42) -> coo_matrix:
    """
    Compute graph connectivities of a nearest neighbors graph based on 
    feature values stored in the  ´obsm´ attribute of an AnnData object. The
    connectivites can be computed exactly from a simple k-nearest-neighbors 
    graph  using sklearn (´mode´ == "knn") or via a fuzzy simplical set based
    on a neighbor graph approximated via nearest neighbor descent (´mode´ == 
    "umap") as in McInnes, L., Healy, J. & Melville, J. UMAP: Uniform Manifold 
    Approximation and Projection for Dimension Reduction. arXiv [stat.ML] (2018).

    Parameters
    ----------
    adata:
        AnnData object with feature values for distance calculation stored in 
        ´adata.obsm[feature_key]´.
    feature_key:
        Key under which the feature values for distance calculation are stored 
        in ´adata.obsm´.
    n_neighbors:
        Number of neighbors for the nearest neighbors graph construction.
    mode:
        If "knn", compute exact connectivities from a knn graph using sklearn.
        If "umap", compute a fuzzy simplical set based on an approximated 
        neighbor graph.
    seed:
        Random seed to get reproducible results.

    Returns
    ----------
    connectivities:
         Sparse matrix that contains the connectivity weights between all cells.
    """ 
    # Get features for graph construction
    X = adata.obsm[feature_key]

    # Compute exact graph connectivities
    if mode == "knn":
        connectivities = kneighbors_graph(X=X, n_neighbors=n_neighbors)
    # Compute graph connectivities using a fuzzy simplical set and approximate
    # neighbor graph
    elif mode == "umap":
        knn_indices, knn_distances = _compute_graph_indices_and_distances(
            adata=adata,
            feature_key=feature_key,
            n_neighbors=n_neighbors,
            mode="umap")

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
