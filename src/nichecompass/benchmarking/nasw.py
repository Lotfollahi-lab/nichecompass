"""
This module contains the Niche Average Silhouette Width (NASW) benchmark
for testing how well the latent feature space can be clustered into distinct
and compact clusters.
"""

from typing import Optional

import numpy as np
import scanpy as sc
import scib_metrics
from anndata import AnnData

from .utils import compute_knn_graph_connectivities_and_distances


def compute_nasw(
        adata: AnnData,
        latent_knng_key: str="nichecompass_latent_knng",
        latent_key: Optional[str]="nichecompass_latent",
        n_neighbors: Optional[int]=15,
        min_res: float=0.1,
        max_res: float=1.0,
        res_num: int=3,
        n_jobs: int=1,
        seed: int=0) -> float:
    """
    Compute the Niche Average Silhouette Width (NASW). The NASW ranges between
    '0' and '1' with higher values indicating more distinct and compact
    clusters in the latent feature space. To compute the NASW, Leiden
    clusterings with different resolutions are computed for the latent nearest
    neighbor graph. The NASW for all clustering resolutions is computed and the
    average value is returned as metric for clusterability.

    If existent, uses a precomputed latent nearest neighbor graph stored in
    ´adata.obsp[latent_knng_key + '_connectivities']´.
    Alternatively, computes it on the fly using ´latent_key´ and ´n_neighbors´,
    and stores it in ´adata.obsp[latent_knng_key + '_connectivities']´.

    Parameters
    ----------
    adata:
        AnnData object with a precomputed latent nearest neighbor graph stored
        in ´adata.obsp[latent_knng_key + '_connectivities']´ or the latent
        representation from a model stored in ´adata.obsm[latent_key]´.
    latent_knng_key:
        Key under which the latent nearest neighbor graph is / will be stored in
        ´adata.obsp´ with the suffix '_connectivities'.
    latent_key:
        Key under which the latent representation from a model is stored in
        ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the construction of the latent nearest
        neighbor graph from the latent representation from a model in case they
        are constructed.
    min_res:
        Minimum resolution for Leiden clustering.
    max_res:
        Maximum resolution for Leiden clustering.
    res_num:
        Number of linearly spaced Leiden resolutions between ´min_res´ and
        ´max_res´ for which Leiden clusterings will be computed.
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    nasw:
        Average NASW across all clustering resolutions.
    """
    # Adding '_connectivities' as expected / added by 
    # 'compute_knn_graph_connectivities_and_distances'
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"
        
    if latent_knng_connectivities_key in adata.obsp:
        print("Using precomputed latent nearest neighbor graph...")
    else:
        print("Computing latent nearest neighbor graph...")
        compute_knn_graph_connectivities_and_distances(
                adata=adata,
                feature_key=latent_key,
                knng_key=latent_knng_key,
                n_neighbors=n_neighbors,
                random_state=seed,
                n_jobs=n_jobs)

    # Define search space of clustering resolutions
    clustering_resolutions = np.linspace(start=min_res,
                                         stop=max_res,
                                         num=res_num,
                                         dtype=np.float32)

    print("Computing latent Leiden clusterings...")
    # Calculate latent Leiden clustering for different resolutions
    for resolution in clustering_resolutions:
        if not f"leiden_latent_{str(resolution)}" in adata.obs:
            sc.tl.leiden(adata,
                         resolution=resolution,
                         random_state=seed,
                         key_added=f"leiden_latent_{str(resolution)}",
                         adjacency=adata.obsp[latent_knng_connectivities_key])
        else:
            print("Using precomputed latent Leiden clusters for resolution "
                  f"{str(resolution)}.")

    print("Computing NASW...")
    # Calculate max MNASW over all clustering resolutions
    nasw_list = []
    for resolution in clustering_resolutions:
        nasw_list.append(scib_metrics.silhouette_label(
            X=adata.obsm[latent_key],
            labels=adata.obs[f"leiden_latent_{str(resolution)}"]))
    nasw = np.mean(nasw_list)
    return nasw