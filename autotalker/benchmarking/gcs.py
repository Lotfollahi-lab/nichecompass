"""
This module contains the Graph Connectivity Similarity (GCS) benchmark for
testing how accurately the latent nearest neighbor graph preserves edges from
the physical (spatial) nearest neighbor graph.
"""

from typing import Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData


def compute_avg_gcs(
        adata: AnnData,
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        min_n_neighbors: int=1,
        max_n_neighbors: int=15,
        seed: int=0) -> float:
    """
    Compute multiple Graph Connectivity Similarities (GCS) by varying the number
    of neighbors used for nearest neighbor graph construction (between
    ´min_n_neighbors´ and ´max_n_neighbors´) and return the average GCS. Can use
    precomputed spatial and latent nearest neighbor graphs stored in
    ´adata.obsp[f'autotalker_spatial_{n_neighbors}nng_connectivities']´ and
    ´adata.obsp[f'autotalker_latent_{n_neighbors}nng_connectivities']´
    respectively.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates stored in
        ´adata.obsm[spatial_key]´ and the latent representation from a model
        stored in ´adata.obsm[latent_key]´. Precomputed nearest neighbor graphs
        can optionally be stored in
        ´adata.obsp[f'autotalker_spatial_{n_neighbors}nng_connectivities']´ and
        ´adata.obsp[f'autotalker_latent_{n_neighbors}nng_connectivities']´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from a model is stored in
        ´adata.obsm´.
    min_n_neighbors:
        Minimum number of neighbors used for computing the average GCS.
    max_n_neighbors:
        Maximum number of neighbors used for computing the average GCS.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    avg_gcs:
        Average GCS computed over different nearest neighbor graphs with varying
        number of neighbors.
    """
    gcs_list = []
    for n_neighbors in range(min_n_neighbors, max_n_neighbors):
        gcs_list.append(compute_gcs(
            adata=adata,
            spatial_knng_key=f"autotalker_spatial_{n_neighbors}nng",
            latent_knng_key=f"autotalker_latent_{n_neighbors}nng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed))
    avg_gcs = np.mean(gcs_list)
    return avg_gcs


def compute_gcs(
        adata: AnnData,
        spatial_knng_key: str="autotalker_spatial_knng",
        latent_knng_key: str="autotalker_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="autotalker_latent",
        n_neighbors: Optional[int]=15,
        seed: Optional[int]=0):
    """
    Compute the graph connectivity similarity (GCS) between the latent nearest
    neighbor graph and the spatial nearest neighbor graph. The GCS measures how
    accurately the latent nearest neighbor graph preserves edges and non-edges
    from the spatial (ground truth) nearest neighbor graph. A value of '1'
    indicates perfect graph similarity and a value of '0' indicates no graph
    similarity at all.
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

    Returns
    ----------
    gcs:
        Normalized matrix similarity between the spatial (ground truth) nearest
        neighbor graph and the latent nearest neighbor graph as measured by one
        minus the size-normalized Frobenius norm of the element-wise matrix
        differences.
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

    # Compute Frobenius norm of the matrix of differences to quantify distance
    connectivities_diff = (adata.obsp[latent_knng_connectivities_key] -
                           adata.obsp[spatial_knng_connectivities_key])
    gcd = sp.linalg.norm(connectivities_diff,
                         ord="fro")
    
    # Normalize gcd to be between 0 and 1 and convert to gcs by subtracting from
    # 1
    gcs = 1 - (gcd / len(adata))
    return gcs
