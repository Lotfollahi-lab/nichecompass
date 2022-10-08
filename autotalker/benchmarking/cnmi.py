import numpy as np
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from sklearn.metrics.cluster import normalized_mutual_info_score


def compute_cluster_nmi(adata: AnnData,
                        seed: int):
    """
    
    """
    # Calculate adjacency matrix for Leiden clustering
    sq.gr.spatial_neighbors(adata,
                            spatial_key="spatial",
                            coord_type="generic",
                            key_added="leiden",
                            n_neighs=6)

    # Calculate Leiden clustering
    clustering_resolutions = np.linspace(0.1, 1.0, 10)
    for resolution in clustering_resolutions:
        sc.tl.leiden(adata,
                     resolution=resolution,
                     random_state=seed,
                     key_added=f"leiden_{resolution}",
                     adjacency=adata.obsp["leiden_connectivities"])


def _nmi(adata, cluster_group_1_key, cluster_group_2_key):
    """
    Calculate the normalized mutual information (NMI) between two different 
    cluster assignments.

    Parameters
    ----------
    adata:
        AnnData object with clustering labels stored in 
        ´adata.obs[cluster_group_1_key]´ and ´adata.obs[cluster_group_2_key]´.
    cluster_group_1_key:
        Key under which the clustering labels from the first clustering 
        assignment are stored in ´adata.obs´.
    cluster_group_2_key:
        Key under which the clustering labels from the second clustering 
        assignment are stored in ´adata.obs´.   

    Returns
    ----------
    nmi_score:
        Normalized mutual information score as calculated by sklearn.
    """

    cluster_group_1 = adata.obs[cluster_group_1_key].tolist()
    cluster_group_2 = adata.obs[cluster_group_2_key].tolist()

    if len(cluster_group_1) != len(cluster_group_2):
        raise ValueError(
            f"Different lengths in cluster_group_1 ({len(cluster_group_1)}) "
            f"and cluster_group_2 ({len(cluster_group_2)})")

    nmi_score = normalized_mutual_info_score(cluster_group_1,
                                             cluster_group_2,
                                             average_method="arithmetic")
    return nmi_score