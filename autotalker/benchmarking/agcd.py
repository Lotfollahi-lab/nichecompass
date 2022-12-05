"""
This module contains a benchmark for testing how good the latent space / latent
neighbor graph preserves spatial information from the original spatial 
coordinates / spatial neighbor graph.
"""

import numpy as np
from anndata import AnnData

from autotalker.utils import compute_graph_connectivities


def compute_avg_gcd(
        adata: AnnData,
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        seed: int=42) -> np.float64:
    """
    Compute multiple graph connectivity distances by varying the number of 
    neighbors used for nearest neighbor graph construction (between 1 and 15)
    and return the average gcd.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in ´adata.obsm[latent_key]´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    seed:
        Random seed to get reproducible results.

    Returns
    ----------
    avg_cad:
        Average graph connectivity distance computed over different nearest 
        neighbor graphs with varying number of neighbors.
    """
    gcd_list = []
    for n_neighbors in range(1,15):
        gcd_list.append(_compute_gcd(
            adata=adata,
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed))
    avg_gcd = np.mean(gcd_list)
    return avg_gcd


def _compute_gcd(
        adata: AnnData,
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        n_neighbors: int=15,
        seed: int=0):
    """
    Compute graph connectivity distance between the latent connectivity graph
    and the spatial connectivity graph.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in ´adata.obsm[latent_key]´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the graph connectivity computation.
    seed:
        Random seed to get reproducible results.

    Returns
    ----------
    graph_connectivity_distance:
        Frobenius norm of matrix differences between the latent connectivity
        graph and the spatial connectivity graph.
    """
    # Compute physical (ground truth) connectivities
    spatial_connectivities = compute_graph_connectivities(
        adata=adata,
        feature_key=spatial_key,
        n_neighbors=n_neighbors,
        mode="knn",
        seed=seed)
    
    # Compute latent connectivities
    latent_connectivites = compute_graph_connectivities(
        adata=adata,
        feature_key=latent_key,
        n_neighbors=n_neighbors,
        mode="knn",
        seed=seed)

    # Calculate Frobenius norm of matrix differences to quantify distance
    connectivities_diff = (latent_connectivites - 
                           spatial_connectivities).toarray()
    graph_connectivity_distance = np.linalg.norm(connectivities_diff,
                                                 ord="fro")
    return graph_connectivity_distance


