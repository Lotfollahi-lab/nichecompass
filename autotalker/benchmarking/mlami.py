"""
This module contains the Maximum Leiden Adjusted Mutual Info (MLAMI) benchmark
for testing how accurately the latent feature space preserves spatial
organization from the physical (spatial) feature space by comparing clustering
overlaps.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import adjusted_mutual_info_score


def compute_mlami(
        adata: AnnData,
        spatial_knng_key: str="autotalker_spatial_knng",
        latent_knng_key: str="autotalker_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="autotalker_latent",
        n_neighbors: Optional[int]=15,
        seed: int=0,
        visualize_leiden_clustering: bool=False) -> float:
    """
    Compute the Maximum Leiden Adjusted Mutual Info (MLAMI) between the latent
    nearest neighbor graph and the spatial nearest neighbor graph. The MLAMI
    ranges between '0' and '1' with higher values indicating that the latent
    feature space more accurately preserves spatial organization from the
    spatial (ground truth) feature space. To compute the MLAMI, Leiden
    clusterings with different resolutions are computed for both nearest
    neighbor graphs. The Adjusted Mutual Info (AMI) between all clustering
    resolution pairs is computed to quantify cluster overlap and the maximum
    value is returned as metric for spatial organization preservation.
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
        a model.
    seed:
        Random seed for reproducibility.
    visualize_leiden_clustering:
        If ´True´, visualize the spatial and latent Leiden clusterings.

    Returns
    ----------
    mlami:
        MLAMI between all clustering resolution pairs.
    """
    # Adding '_connectivities' as automatically added by sc.pp.neighbors
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_knng_connectivities_key not in adata.obsp:
        # Compute spatial (ground truth) connectivities
        sc.pp.neighbors(adata=adata,
                        use_rep=spatial_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=spatial_knng_key)

    if latent_knng_connectivities_key not in adata.obsp:
        # Compute latent connectivities
        sc.pp.neighbors(adata=adata,
                        use_rep=latent_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=latent_knng_key)

    # Define search space of clustering resolutions
    clustering_resolutions = np.linspace(start=0.1,
                                         stop=1.0,
                                         num=10,
                                         dtype=np.float32)

    # Calculate spatial Leiden clustering for different resolutions
    for resolution in clustering_resolutions:
        sc.tl.leiden(adata=adata,
                     resolution=resolution,
                     random_state=seed,
                     key_added=f"leiden_spatial_{str(resolution)}",
                     adjacency=adata.obsp[spatial_knng_connectivities_key])

    # Plot Leiden clustering
    if visualize_leiden_clustering:
        with plt.rc_context({"figure.figsize": (5, 5)}):
            sc.pl.spatial(adata=adata,
                          color=[f"leiden_spatial_{str(resolution)}" for
                                 resolution in clustering_resolutions],
                          ncols=5,
                          spot_size=0.03,
                          legend_loc=None)

    # Calculate latent Leiden clustering for different resolutions
    for resolution in clustering_resolutions:
        sc.tl.leiden(adata,
                     resolution=resolution,
                     random_state=seed,
                     key_added=f"leiden_latent_{str(resolution)}",
                     adjacency=adata.obsp[latent_knng_connectivities_key])
            
    # Plot Leiden clustering
    if visualize_leiden_clustering:
        with plt.rc_context({"figure.figsize": (5, 5)}):
            sc.pl.spatial(adata,
                          color=[f"leiden_latent_{str(resolution)}" for
                                 resolution in clustering_resolutions],
                          ncols=5,
                          spot_size=0.03,
                          legend_loc=None)

    # Calculate max LAMI over all clustering resolutions
    lami_list = []
    for spatial_resolution in clustering_resolutions:
        for latent_resolution in clustering_resolutions:
            lami_list.append(_compute_ami(
                adata=adata,
                cluster_group1_key=f"leiden_spatial_{str(spatial_resolution)}",
                cluster_group2_key=f"leiden_latent_{str(latent_resolution)}"))
    mlami = np.max(lami_list)
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
