"""
This module contains the Maximum Leiden Adjusted Mutual Info (MLAMI) benchmark
for testing how accurately the latent feature space preserves global spatial
organization from the spatial (physical) feature space by comparing clustering
overlaps.
"""

from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import adjusted_mutual_info_score

from .utils import compute_knn_graph_connectivities_and_distances


def compute_mlami(
        adata: AnnData,
        batch_key: Optional[str]=None,
        spatial_knng_key: str="spatial_knng",
        latent_knng_key: str="nichecompass_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="nichecompass_latent",
        n_neighbors: Optional[int]=15,
        min_res: float=0.1,
        max_res: float=1.0,
        res_num: int=3,
        n_jobs: int=1,
        seed: int=0) -> float:
    """
    Compute the Maximum Leiden Adjusted Mutual Info (MLAMI). The MLAMI ranges
    between '0' and '1' with higher values indicating that the latent feature
    space more accurately preserves global spatial organization from the spatial
    (ground truth) feature space. To compute the MLAMI, Leiden clusterings with
    different resolutions are computed for both nearest neighbor graphs. The
    Adjusted Mutual Info (AMI) between all clustering resolution pairs is
    computed to quantify cluster overlap and the maximum value is returned as
    metric for spatial organization preservation.

    If a ´batch_key´ is provided, the MLAMI will be computed on each batch
    separately (with latent Leiden clusters computed on the integrated latent
    space), and the average across all batches is returned.

    If existent, uses precomputed nearest neighbor graphs stored in
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´.
    Alternatively, computes them on the fly using ´spatial_key´, ´latent_key´
    and ´n_neighbors´, and stores them in 
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´ respectively.

    Parameters
    ----------
    adata:
        AnnData object with precomputed nearest neighbor graphs stored in
        ´adata.obsp[spatial_knng_key + '_connectivities']´ and
        ´adata.obsp[latent_knng_key + '_connectivities']´ or spatial coordinates
        stored in ´adata.obsm[spatial_key]´ and the latent representation from a
        model stored in ´adata.obsm[latent_key]´.
    batch_key:
        Key under which the batches are stored in ´adata.obs´. If ´None´, the
        adata is assumed to only have one unique batch.
    spatial_knng_key:
        Key under which the spatial nearest neighbor graph is / will be stored
        in ´adata.obsp´ with the suffix '_connectivities'.
    latent_knng_key:
        Key under which the latent nearest neighbor graph is / will be stored in
        ´adata.obsp´ with the suffix '_connectivities'.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from a model is stored in
        ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the construction of the nearest neighbor
        graphs from the spatial coordinates and the latent representation from
        a model in case they are constructed.
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
    mlami:
        MLAMI between all clustering resolution pairs.
    """
    # Adding '_connectivities' as expected / added by 
    # 'compute_knn_graph_connectivities_and_distances'
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if batch_key is not None:
        adata_batch_list = []
        unique_batches = adata.obs[batch_key].unique().tolist()
        for batch in unique_batches:
            adata_batch = adata[adata.obs[batch_key] == batch]
            adata_batch_list.append(adata_batch)       

    if spatial_knng_connectivities_key in adata.obsp:
        print("Using precomputed spatial nearest neighbor graph...")
    elif batch_key is None:
        print("Computing spatial nearest neighbor graph for entire dataset...")
        compute_knn_graph_connectivities_and_distances(
                adata=adata,
                feature_key=spatial_key,
                knng_key=spatial_knng_key,
                n_neighbors=n_neighbors,
                random_state=seed,
                n_jobs=n_jobs)
    elif batch_key is not None:
        # Compute spatial nearest neighbor graph for each batch separately
        for i, batch in enumerate(unique_batches):
            print("Computing spatial nearest neighbor graph for "
                  f"{batch_key} {batch}...")
            compute_knn_graph_connectivities_and_distances(
                    adata=adata_batch_list[i],
                    feature_key=spatial_key,
                    knng_key=spatial_knng_key,
                    n_neighbors=n_neighbors,
                    random_state=seed,
                    n_jobs=n_jobs)
        
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

    if batch_key is None:
        print("Computing spatial Leiden clusterings for entire dataset...")
        # Calculate spatial Leiden clustering for different resolutions
        for resolution in clustering_resolutions:
            sc.tl.leiden(adata=adata,
                        resolution=resolution,
                        random_state=seed,
                        key_added=f"leiden_spatial_{str(resolution)}",
                        adjacency=adata.obsp[spatial_knng_connectivities_key])
    elif batch_key is not None:
        # Compute spatial Leiden clustering for each batch separately
        for i, batch in enumerate(unique_batches):
            print("Computing spatial Leiden clusterings for "
                  f"{batch_key} {batch}...")
            # Calculate spatial Leiden clustering for different resolutions
            for resolution in clustering_resolutions:
                sc.tl.leiden(
                    adata=adata_batch_list[i],
                    resolution=resolution,
                    random_state=seed,
                    key_added=f"leiden_spatial_{str(resolution)}",
                    adjacency=adata_batch_list[i].obsp[spatial_knng_connectivities_key])        

    print("Computing latent Leiden clusterings...")
    # Calculate latent Leiden clustering for different resolutions
    for resolution in clustering_resolutions:
        sc.tl.leiden(adata,
                     resolution=resolution,
                     random_state=seed,
                     key_added=f"leiden_latent_{str(resolution)}",
                     adjacency=adata.obsp[latent_knng_connectivities_key])
        if batch_key is not None:
            for i, batch in enumerate(unique_batches):
                adata_batch_list[i].obs[f"leiden_latent_{str(resolution)}"] = (
                    adata.obs[f"leiden_latent_{str(resolution)}"])

    if batch_key is None:
        print("Computing MLAMI for entire dataset...")
        # Calculate max LAMI over all clustering resolutions
        lami_list = []
        for spatial_resolution in clustering_resolutions:
            for latent_resolution in clustering_resolutions:
                lami_list.append(_compute_ami(
                    adata=adata,
                    cluster_group1_key=f"leiden_spatial_{str(spatial_resolution)}",
                    cluster_group2_key=f"leiden_latent_{str(latent_resolution)}"))
        mlami = np.max(lami_list)
    elif batch_key is not None:
        for i, batch in enumerate(unique_batches):
            print(f"Computing MLAMI for {batch_key} {batch}...")
            batch_lami_list = []
            for spatial_resolution in clustering_resolutions:
                for latent_resolution in clustering_resolutions:
                    batch_lami_list.append(_compute_ami(
                        adata=adata_batch_list[i],
                        cluster_group1_key=f"leiden_spatial_{str(spatial_resolution)}",
                        cluster_group2_key=f"leiden_latent_{str(latent_resolution)}"))
            batch_mlami = np.max(batch_lami_list)
        mlami = np.mean(batch_mlami)
    return mlami


def _compute_ami(adata: AnnData,
                 cluster_group1_key: str,
                 cluster_group2_key: str) -> float:
    """
    Compute the Adjusted Mutual Information (AMI) between two different
    cluster assignments. AMI compares the overlap of two clusterings. For
    details, see documentation at
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score.

    Parameters
    ----------
    adata:
        AnnData object with clustering labels stored in
        ´adata.obs[cluster_group1_key]´ and ´adata.obs[cluster_group2_key]´.
    cluster_group1_key:
        Key under which the clustering labels from the first clustering
        assignment are stored in ´adata.obs´.
    cluster_group2_key:
        Key under which the clustering labels from the second clustering
        assignment are stored in ´adata.obs´.

    Returns
    ----------
    ami:
        AMI score as calculated by the sklearn implementation.
    """
    cluster_group1 = adata.obs[cluster_group1_key].tolist()
    cluster_group2 = adata.obs[cluster_group2_key].tolist()

    ami = adjusted_mutual_info_score(cluster_group1,
                                     cluster_group2,
                                     average_method="arithmetic")
    return ami
