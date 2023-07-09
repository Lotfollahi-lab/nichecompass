"""
This module contains utilities to compute nearest neighbor graphs for use by the
NicheCompass model.
"""

import scanpy as sc
from anndata import AnnData
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
    sp_distances, sp_conns = sc.neighbors._compute_connectivities_umap(
            indices[:, :n_neighbors],
            distances[:, :n_neighbors],
            adata.n_obs,
            n_neighbors=n_neighbors)
    adata.obsp[f"{knng_key}_connectivities"] = sp_conns
    adata.obsp[f"{knng_key}_distances"] = sp_distances
    adata.uns[f"{knng_key}_n_neighbors"] = n_neighbors