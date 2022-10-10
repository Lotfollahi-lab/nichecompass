from typing import Optional

import numpy as np
from anndata import AnnData
from sklearn.metrics import pairwise_distances


def _compute_knn_graph(adata: AnnData,
                       feature_key: str,
                       n_neighbors: int=6):
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

    Returns
    ----------
    knn_indices:
        NumPy array that contains the indices of the ´n_neighbors´ nearest 
        neighbors for each observation.
    knn_distances:
        NumPy array that contains the distances to the ´n_neighbors´ nearest 
        neighbors for each observation.
    """
    distances = pairwise_distances(X=adata.obsm[feature_key],
                                      metric="euclidean")

    # Retrieve knn indices and knn distances from pairwise distances
    obs_range = np.arange(distances.shape[0])[:, None]
    knn_indices = np.argpartition(distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
    knn_indices = knn_indices[obs_range, np.argsort(distances[obs_range,
                                                    knn_indices])]
    knn_distances = distances[obs_range, knn_indices]
    return knn_indices, knn_distances
    

def _convert_to_one_hot(vector: np.ndarray,
                        n_classes: Optional[int]):
    """
    Converts an input 1D vector of integers into a 2D array of one-hot vectors,
    where for an i'th input value of j, a '1' will be inserted in the i'th row, 
    j'th column of the output one-hot vector. Adapted from 
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
        One-hot-encoded vector.

    Example:
    ´´´
    vector = np.array((1, 0, 4))
    one_hot_vector = _convert_to_one_hot(vector)
    print(one_hot_vector)
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
