"""
This module contains the Graph Connectivity Distance (GCD) benchmark for testing
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
    Compute multiple Graph Connectivity Distances (GCDs) by varying the number
    of neighbors used for nearest neighbor graph construction (between
    ´min_n_neighbors´ and ´max_n_neighbors´) and return the average GCD. Can use
    precomputed spatial and latent nearest neighbor graphs stored in
    ´adata.obsp[f'autotalker_spatial_{n_neighbors}nng_connectivities']´ and
    ´adata.obsp[f'autotalker_latent_{n_neighbors}nng_connectivities']´
    respectively.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates stored in
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in ´adata.obsm[latent_key]´. Precomputed nearest neighbor graphs
        can optionally be stored in
        ´adata.obsp[f'autotalker_spatial_{n_neighbors}nng_connectivities']´ and
        ´adata.obsp[f'autotalker_latent_{n_neighbors}nng_connectivities']´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from the model is stored in
        ´adata.obsm´.
    min_n_neighbors:
        Minimum number of neighbors used for computing the average GCD.
    max_n_neighbors:
        Maximum number of neighbors used for computing the average GCD.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    avg_cad:
        Average GCD computed over different nearest neighbor graphs with varying
        number of neighbors.
    """
    gcd_list = []
    for n_neighbors in range(min_n_neighbors, max_n_neighbors):
        gcd_list.append(compute_gcd(
            adata=adata,
            spatial_knng_key=f"autotalker_spatial_{n_neighbors}nng",
            latent_knng_key=f"autotalker_latent_{n_neighbors}nng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed))
    avg_gcd = np.mean(gcd_list)
    return avg_gcd


def compute_gcd(
        adata: AnnData,
        spatial_knng_key: str="autotalker_spatial_8nng",
        latent_knng_key: str="autotalker_latent_8nng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="autotalker_latent",
        n_neighbors: Optional[int]=8,
        seed: Optional[int]=0):
    """
    Compute the graph connectivity distance (GCD) between the latent nearest
    neighbor graph and the spatial nearest neighbor graph. A lower value
    indicates a latent nearest neighbor graph that more accurately preserves
    edges from the spatial (ground truth) nearest neighbor graph.
    If existent, use precomputed nearest neighbor graphs stored in
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´.
    Alternatively, compute them on the fly using ´spatial_key´, ´latent_key´ and
    ´n_neighbors´.

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

    Returns
    ----------
    gcd:
        Matrix distance between the spatial (ground truth) nearest neighbor
        graph and the latent nearest neighbor graph as measured by the Frobenius
        norm of the element-wise matrix differences.
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

    # Compute Frobenius norm of the matrix of differences to quantify distance
    connectivities_diff = (adata.obsp[latent_knng_connectivities_key] -
                           adata.obsp[spatial_knng_connectivities_key]
                           ).toarray()
    gcd = np.linalg.norm(connectivities_diff,
                         ord="fro")
    return gcd
