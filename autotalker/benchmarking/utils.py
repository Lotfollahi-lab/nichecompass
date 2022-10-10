import numpy as np
from anndata import AnnData
from sklearn.metrics import pairwise_distances


def _compute_knn_graph(adata: AnnData,
                       feature_key: str,
                       n_neighbors: int=6):
    """
    Compute a k-nearest-neighbors graph based on feature values stored in an
    AnnData object and return the indices of and distances to the k-nearest 
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
    
