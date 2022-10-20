import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from sklearn.metrics import normalized_mutual_info_score

from .utils import _compute_graph_connectivities


def compute_max_lnmi(
        adata: AnnData,
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        n_neighbors: int=15,
        seed: int=42,
        visualize_leiden_clustering: bool=False):
    """
    Compute the maximum Leiden Normalized Mutual Info (LNMI). First, graph
    connectivites are computed from the spatial coordinates (ground truth) and
    from the latent representation of the model (latent) respectively.
    Leiden clusterings with different resolutions are computed for both nearest
    neighbor graphs. The NMI between all clustering resolution pairs is
    calculated to quantify cluster overlap and the maximum value is chosen as
    a metric for spatial tissue organization preservation.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in adata.obsm[latent_key].
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
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
        Random seed to get reproducible results.
    visualize_leiden_clustering:
        If ´True´, also visualize the spatial/physical and latent Leiden 
        clusterings.

    Returns
    ----------
    max_lnmi:
        Maximum cluster overlap between all resolution pairs.
    """
    clustering_resolutions = np.linspace(start=0.1,
                                         stop=1.0,
                                         num=10,
                                         dtype=np.float32)

    # Compute physical (ground truth) connectivities
    adata.obsp["lnmi_spatial_connectivities"] = _compute_graph_connectivities(
        adata=adata,
        feature_key=spatial_key,
        n_neighbors=n_neighbors,
        mode="knn",
        seed=seed)

    # Calculate spatial Leiden clustering for different resolutions
    for resolution in clustering_resolutions:
        sc.tl.leiden(adata,
                     resolution=resolution,
                     random_state=seed,
                     key_added=f"leiden_spatial_{str(resolution)}",
                     adjacency=adata.obsp["lnmi_spatial_connectivities"])

    # Plot Leiden clustering
    if visualize_leiden_clustering:
        with plt.rc_context({"figure.figsize": (5, 5)}):
            sc.pl.spatial(adata,
                          color=[f"leiden_spatial_{str(resolution)}" for resolution in clustering_resolutions],
                          ncols=5,
                          spot_size=0.03,
                          legend_loc=None)

    # Compute latent connectivities
    adata.obsp["lnmi_latent_connectivities"] = _compute_graph_connectivities(
        adata=adata,
        feature_key=latent_key,
        n_neighbors=n_neighbors,
        mode="knn",
        seed=seed)

    # Calculate latent Leiden clustering for different resolutions
    for resolution in clustering_resolutions:
        sc.tl.leiden(adata,
                     resolution=resolution,
                     random_state=seed,
                     key_added=f"leiden_latent_{str(resolution)}",
                     adjacency=adata.obsp["lnmi_latent_connectivities"])
                
    # Plot Leiden clustering
    if visualize_leiden_clustering:
        with plt.rc_context({"figure.figsize": (5, 5)}):
            sc.pl.spatial(adata,
                          color=[f"leiden_latent_{str(resolution)}" for resolution in clustering_resolutions],
                          ncols=5,
                          spot_size=0.03,
                          legend_loc=None)

    # Calculate max lnmi over all clustering resolutions
    lnmi_list = []
    for spatial_resolution in clustering_resolutions:
        for latent_resolution in clustering_resolutions:
            lnmi_list.append(_compute_nmi(adata,
                             f"leiden_spatial_{str(spatial_resolution)}",
                             f"leiden_latent_{str(latent_resolution)}"))

    max_lnmi = np.max(lnmi_list)
    return max_lnmi


def _compute_nmi(adata: AnnData,
                 cluster_group1_key: str,
                 cluster_group2_key: str):
    """
    Calculate the normalized mutual information (NMI) between two different 
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
        Normalized mutual information score as calculated by sklearn.
    """
    cluster_group1 = adata.obs[cluster_group1_key].tolist()
    cluster_group2 = adata.obs[cluster_group2_key].tolist()

    if len(cluster_group1) != len(cluster_group2):
        raise ValueError(
            f"Different lengths in cluster_group1 ({len(cluster_group1)}) "
            f"and cluster_group2 ({len(cluster_group2)})")

    nmi = normalized_mutual_info_score(cluster_group1,
                                       cluster_group2,
                                       average_method="arithmetic")
    return nmi