import numpy as np
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from sklearn.metrics import normalized_mutual_info_score


def compute_min_lnmi_metric(
        adata: AnnData,
        spatial_key: str="spatial",
        latent_rep_key: str="autotalker_latent",
        neighborhood_graph_n_neighs: int=6,
        seed: int=42):
    """
    
    """
    clustering_resolutions = np.linspace(start=0.1,
                                         stop=1.0,
                                         num=10,
                                         dtype=np.float32)

    # Create neighbor graph from spatial coordinates
    sq.gr.spatial_neighbors(adata,
                            spatial_key=spatial_key, 
                            coord_type="generic",
                            n_neighs=neighborhood_graph_n_neighs,
                            key_added="lnmi_spatial")

    # Calculate spatial Leiden clustering for different resolutions
    for resolution in clustering_resolutions:
        sc.tl.leiden(adata,
                     resolution=resolution,
                     random_state=seed,
                     key_added=f"lnmi_spatial_{str(resolution)}",
                     adjacency=adata.obsp["lnmi_spatial_connectivities"])

    # Create neighbor graph from latent representation
    """
    sc.pp.neighbors did not give expected results
    sc.pp.neighbors(adata,
                    n_neighbors=neighborhood_graph_n_neighs,
                    use_rep=latent_rep_key,
                    random_state=seed,
                    key_added="lnmi_latent")
    """

    sq.gr.spatial_neighbors(adata,
                            spatial_key=latent_rep_key,
                            coord_type="generic",
                            n_neighs=neighborhood_graph_n_neighs,
                            key_added="lnmi_latent")

    # Calculate adjacency matrix for latent Leiden clustering
    for resolution in clustering_resolutions:
        sc.tl.leiden(adata,
                     resolution=resolution,
                     random_state=seed,
                     key_added=f"lnmi_latent_{str(resolution)}",
                     adjacency=adata.obsp["lnmi_latent_connectivities"])

    nmi_list = []

    for spatial_resolution in clustering_resolutions:
        for latent_resolution in clustering_resolutions:
            nmi_list.append(_nmi(adata,
                            f"lnmi_spatial_{str(spatial_resolution)}",
                            f"lnmi_latent_{str(latent_resolution)}"))

    min_nmi = np.min(nmi_list)
    return min_nmi


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