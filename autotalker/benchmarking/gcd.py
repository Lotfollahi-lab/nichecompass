"""
This module contains the graph connectivity distance (GCD) benchmark for testing
how good the latent nearest neighbor graph preserves edges from the original 
spatial nearest neighbor graph.
"""

from typing import Optional

import numpy as np
from anndata import AnnData

from autotalker.utils import compute_graph_connectivities


def compute_avg_gcd(
        adata: AnnData,
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        min_n_neighbors: int=1,
        max_n_neighbors: int=16,
        seed: int=0) -> float:
    """
    Compute multiple graph connectivity distances by varying the number of 
    neighbors used for nearest neighbor graph construction (between 
    ´min_n_neighbors´ and ´max_n_neighbors´) and return the average graph 
    connectivity distance.

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
    min_n_neighbors:
        Minimum number of neighbors used for computing the average graph
        connectivity distance.
    max_n_neighbors:
        Maximum number of neighbors used for computing the average graph
        connectivity distance.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    avg_cad:
        Average graph connectivity distance computed over different nearest 
        neighbor graphs with varying number of neighbors.
    """
    gcd_list = []
    for n_neighbors in range(min_n_neighbors, max_n_neighbors):
        gcd_list.append(compute_gcd(
            adata=adata,
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed))
    avg_gcd = np.mean(gcd_list)
    return avg_gcd


def compute_gcd(
        adata: AnnData,
        spatial_connectivities_key: str="autotalker_spatial_connectivities",
        latent_connectivities_key: str="autotalker_latent_connectivities",
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        n_neighbors: int=8,
        seed: int=0):
    """
    Compute the graph connectivity distance between the latent nearest neighbor 
    graph and the spatial nearest neighbor graph. Use precomputed nearest
    neighbor graphs stored in ´adata.obsp[]´ or compute them on the fly using ´spatial_key´ and
    ´latent_key´.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in ´adata.obsm[latent_key]´.
        spatial_connectivities_key:
            Key under which the spatial nearest neighbor graph is / will be 
            stored in ´adata.obsp´.       
        latent_connectivities_key:
            Key under which the latent nearest neighbor graph is / will be
            stored in ´adata.obsp´.  
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

    Returns
    ----------
    graph_connectivity_distance:
        Frobenius norm of the matrix of differences between the latent nearest
        neighbor graph and the spatial nearest neighbor graph.
    """
    if spatial_connectivities_key not in adata.obsp:
        # Compute spatial (ground truth) connectivities
        adata.obsp[spatial_connectivities_key] = compute_graph_connectivities(
            adata=adata,
            feature_key=spatial_key,
            n_neighbors=n_neighbors,
            mode="knn",
            seed=seed)
    
    if latent_connectivities_key not in adata.obsp:
        # Compute latent connectivities
        adata.obsp[latent_connectivities_key] = compute_graph_connectivities(
            adata=adata,
            feature_key=latent_key,
            n_neighbors=n_neighbors,
            mode="knn",
            seed=seed)

    # Calculate Frobenius norm of matrix differences to quantify distance
    connectivities_diff = (adata.obsp[latent_connectivities_key] - 
                           adata.obsp[spatial_connectivities_key]).toarray()
    graph_connectivity_distance = np.linalg.norm(connectivities_diff,
                                                 ord="fro")
    return graph_connectivity_distance


