"""
This module contains the Maximum Leiden Normalized Mutual Info (MLNMI) benchmark
for testing how accurately the latent feature space preserves spatial
organization from the original spatial feature space by comparing clustering
overlaps.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import normalized_mutual_info_score

from autotalker.utils import compute_graph_connectivities


def compute_mlnmi(
        adata: AnnData,
        spatial_knng_key: Optional[str]="autotalker_spatial_8nng",
        latent_knng_key: Optional[str]="autotalker_latent_8nng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="autotalker_latent",
        n_neighbors: Optional[int]=8,
        seed: Optional[int]=0,
        visualize_leiden_clustering: bool=False):
    """
    Compute the Maximum Leiden Normalized Mutual Info (MLNMI) between the latent
    nearest neighbor graph and the spatial nearest neighbor graph. A higher
    value indicates that the latent feature space more accurately preserves 
    spatial organization from the original spatial feature space. If existent,
    use precomputed nearest neighbor graphs stored in
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´.
    Alternatively, compute them on the fly using ´spatial_key´, ´latent_key´ and
    ´n_neighbors´. Leiden clusterings with different resolutions are computed
    for both nearest neighbor graphs. The Normalized Mutual Info (NMI) between
    all clustering resolution pairs is computed to quantify cluster overlap and
    the maximum value is returned as metric for spatial organization
    preservation.

    Parameters
    ----------
    adata:
        AnnData object with precomputed nearest neighbor graphs stored in
        ´adata.obsp[spatial_knng_key + '_connectivities']´ and
        ´adata.obsp[latent_knng_key + '_connectivities']´ or, alternatively,
        spatial coordinates stored in ´adata.obsm[spatial_key]´ and the latent
        representation from the model stored in ´adata.obsm[latent_key]´.
    spatial_knng_key:
        Key under which the spatial nearest neighbor graph is / will be stored
        in ´adata.obsp´ with the suffix '_connectivities'.
    latent_knng_key:
        Key under which the latent nearest neighbor graph is / will be stored in
        ´adata.obsp´ with the suffix '_connectivities'.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from the model is stored in
        ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the construction of the nearest neighbor
        graphs from the spatial coordinates and the latent representation from
        the model.
    seed:
        Random seed for reproducibility.
    visualize_leiden_clustering:
        If ´True´, visualize the spatial and latent Leiden clusterings.

    Returns
    ----------
    max_lnmi:
        MLNMI between all clustering resolution pairs.
    """
    # Adding '_connectivities' as required by squidpy
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_knng_connectivities_key not in adata.obsp:
        # Compute spatial (ground truth) connectivities
        adata.obsp[spatial_knng_connectivities_key] = (
            compute_graph_connectivities(
                adata=adata,
                feature_key=spatial_key,
                n_neighbors=n_neighbors,
                mode="knn",
                seed=seed))

    if latent_knng_connectivities_key not in adata.obsp:
        # Compute latent connectivities
        adata.obsp[latent_knng_connectivities_key] = (
            compute_graph_connectivities(
                adata=adata,
                feature_key=latent_key,
                n_neighbors=n_neighbors,
                mode="knn",
                seed=seed))

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

    # Calculate max lnmi over all clustering resolutions
    lnmi_list = []
    for spatial_resolution in clustering_resolutions:
        for latent_resolution in clustering_resolutions:
            lnmi_list.append(_compute_nmi(
                adata=adata,
                cluster_group1_key=f"leiden_spatial_{str(spatial_resolution)}",
                cluster_group2_key=f"leiden_latent_{str(latent_resolution)}"))
    max_lnmi = np.max(lnmi_list)
    return max_lnmi


def _compute_nmi(adata: AnnData,
                 cluster_group1_key: str,
                 cluster_group2_key: str):
    """
    Compute the Normalized Mutual Information (NMI) between two different
    cluster assignments. NMI compares the overlap of two clusterings.

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
    nmi:
        NMI score as calculated by sklearn.
    """
    cluster_group1 = adata.obs[cluster_group1_key].tolist()
    cluster_group2 = adata.obs[cluster_group2_key].tolist()

    # Validate cluster groups
    if len(cluster_group1) != len(cluster_group2):
        raise ValueError(
            f"Different lengths in 'cluster_group1' ({len(cluster_group1)}) "
            f"and 'cluster_group2' ({len(cluster_group2)}).")

    nmi = normalized_mutual_info_score(cluster_group1,
                                       cluster_group2,
                                       average_method="arithmetic")
    return nmi
