"""
This module contains utilities to compute nearest neighbor graphs for use by the
NicheCompass model.
"""

import scanpy as sc
from anndata import AnnData
import numpy as np
from scib_metrics.nearest_neighbors import pynndescent


def compute_knn_graph_connectivities_and_distances(
        adata: AnnData,
        feature_key: str="nichecompass_latent",
        knng_key: str="nichecompass_latent_15knng",
        n_neighbors: int=15,
        random_state: int=0,
        n_jobs: int=1) -> None:
    """
    Compute approximate k-nearest-neighbors graph, and add connectivities and
    distances to the adata object.

    Parameters
    ----------
    adata:
        AnnData object with the features for knn graph computation stored in
        ´adata.obsm[feature_key]´.
    feature_key:
        Key in ´adata.obsm´ that will be used to compute the knn graph.
    knng_key:
        Key under which the knn graph connectivities  will be stored
        in ´adata.obsp´ with the suffix '_connectivities', the knn graph
        distances will be stored in ´adata.obsp´ with the suffix '_distances',
        and the number of neighbors will be stored in ´adata.uns with the suffix
        '_n_neighbors' .      
    n_neighbors:
        Number of neighbors of the knn graph.
    random_state:
        Random state for reproducibility.   
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    """
    neigh_output = pynndescent(
        adata.obsm[feature_key],
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=n_jobs)
    indices, distances = neigh_output.indices, neigh_output.distances
    
    # This is a trick to get lisi metrics to work by adding the tiniest possible value
    # to 0 distance neighbors so that each cell has the same amount of neighbors 
    # (otherwise some cells lose neighbors with distance 0 due to sparse representation)
    row_idx = np.where(distances == 0)[0]
    col_idx = np.where(distances == 0)[1]
    new_row_idx = row_idx[np.where(row_idx != indices[row_idx, col_idx])[0]]
    new_col_idx = col_idx[np.where(row_idx != indices[row_idx, col_idx])[0]]
    distances[new_row_idx, new_col_idx] = (distances[new_row_idx, new_col_idx] +
                                           np.nextafter(0, 1, dtype=np.float32))

    sp_distances, sp_conns = sc.neighbors._compute_connectivities_umap(
            indices[:, :n_neighbors],
            distances[:, :n_neighbors],
            adata.n_obs,
            n_neighbors=n_neighbors)
    adata.obsp[f"{knng_key}_connectivities"] = sp_conns
    adata.obsp[f"{knng_key}_distances"] = sp_distances
    adata.uns[f"{knng_key}_n_neighbors"] = n_neighbors