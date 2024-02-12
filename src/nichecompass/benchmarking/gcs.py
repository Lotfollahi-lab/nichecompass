"""
This module contains the Graph Connectivity Similarity (GCS) benchmark for
testing how accurately the latent nearest neighbor graph preserves edges and
non-edges from the spatial (physical) nearest neighbor graph.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .utils import compute_knn_graph_connectivities_and_distances


def compute_gcs(
        adata: AnnData,
        batch_key: Optional[str]=None,
        spatial_knng_key: str="spatial_knng",
        latent_knng_key: str="nichecompass_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="nichecompass_latent",
        n_neighbors: Optional[int]=15,
        n_jobs: int=1,
        seed: int=0):
    """
    Compute the graph connectivity similarity (GCS). The GCS measures how
    accurately the latent nearest neighbor graph preserves edges and non-edges
    from the spatial (ground truth) nearest neighbor graph. A value of '1'
    indicates perfect graph similarity and a value of '0' indicates no graph
    connectivity similarity at all.

    If a ´batch_key´ is provided, the GCS will be computed on each batch
    separately, and the average across all batches is returned.

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
        Key under which the batches are stored in ´adata.obs´.
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
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    gcs:
        Normalized matrix similarity between the spatial nearest neighbor graph
        and the latent nearest neighbor graph as measured by one minus the
        size-normalized Frobenius norm of the element-wise matrix differences.
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
    elif batch_key is None:
        print("Computing latent nearest neighbor graph for entire dataset...")
        compute_knn_graph_connectivities_and_distances(
                adata=adata,
                feature_key=latent_key,
                knng_key=latent_knng_key,
                n_neighbors=n_neighbors,
                random_state=seed,
                n_jobs=n_jobs)
    elif batch_key is not None:
        # Compute latent nearest neighbor graph for each batch separately
        for i, batch in enumerate(unique_batches):
            print("Computing latent nearest neighbor graph for "
                  f"{batch_key} {batch}...")
            compute_knn_graph_connectivities_and_distances(
                    adata=adata_batch_list[i],
                    feature_key=latent_key,
                    knng_key=latent_knng_key,
                    n_neighbors=n_neighbors,
                    random_state=seed,
                    n_jobs=n_jobs)

    if batch_key is None:
        print("Computing GCS for entire dataset...")
        n_neighbors = adata.uns[f"{latent_knng_key}_n_neighbors"]
        # Compute Frobenius norm of the matrix of differences to quantify
        # distance (square root of the sum of absolute squares)
        connectivities_diff = (
            adata.obsp[latent_knng_connectivities_key] -
            adata.obsp[spatial_knng_connectivities_key])
        gcd = sp.linalg.norm(connectivities_diff,
                             ord="fro")
        
        # Normalize gcd to be between 0 and 1 and convert to gcs by subtracting
        # from 1. Maximum number of differences per node is 2 * n_neighbors (
        # sc.pp.neighbors returns a weighted symmetric knn graph with the node-
        # wise sums of weights not exceeding the number of neighbors; the
        # maximum difference for a node is reached if none of the neighbors
        # coincide for the node)
        gcs = 1 - (gcd ** 2 / (n_neighbors * 2 * connectivities_diff.shape[0]))
    elif batch_key is not None:
        # Compute GCS per batch and average
        gcs_list = []
        for i, batch in enumerate(unique_batches):
            print(f"Computing GCS for {batch_key} {batch}...")
            n_neighbors = adata_batch_list[i].uns[
                f"{latent_knng_key}_n_neighbors"]
            batch_connectivities_diff = (
                adata_batch_list[i].obsp[latent_knng_connectivities_key] -
                adata_batch_list[i].obsp[spatial_knng_connectivities_key])
            batch_gcd = sp.linalg.norm(batch_connectivities_diff,
                                       ord="fro")
            batch_gcs = 1 - (
                batch_gcd ** 2 / (
                n_neighbors * 2 * batch_connectivities_diff.shape[0]))
            gcs_list.append(batch_gcs)
        gcs = np.mean(gcs_list)
    return gcs


def compute_avg_gcs(
        adata: AnnData,
        batch_key: Optional[str]=None,
        spatial_key: str="spatial",
        latent_key: str="nichecompass_latent",
        min_n_neighbors: int=1,
        max_n_neighbors: int=15,
        seed: int=0) -> float:
    """
    Compute multiple Graph Connectivity Similarities (GCS) by varying the number
    of neighbors used for nearest neighbor graph construction (between
    ´min_n_neighbors´ and ´max_n_neighbors´) and return the average GCS. Can use
    precomputed spatial and latent nearest neighbor graphs stored in
    ´adata.obsp[f'{spatial_key}_{n_neighbors}nng_connectivities']´ and
    ´adata.obsp[f'{latent_key}_{n_neighbors}nng_connectivities']´ respectively.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates stored in
        ´adata.obsm[spatial_key]´ and the latent representation from a model
        stored in ´adata.obsm[latent_key]´. Precomputed nearest neighbor graphs
        can optionally be stored in
        ´adata.obsp[f'{spatial_key}_{n_neighbors}nng_connectivities']´
        and ´adata.obsp[f'{latent_key}_{n_neighbors}nng_connectivities']´.
    batch_key:
        Key under which the batches are stored in ´adata.obs´. If ´None´, the
        adata is assumed to only have one unique batch.
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
            batch_key=batch_key,
            spatial_knng_key=f"{spatial_key}_{n_neighbors}nng",
            latent_knng_key=f"{latent_key}_{n_neighbors}nng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed))
    avg_gcs = np.mean(gcs_list)
    return avg_gcs
