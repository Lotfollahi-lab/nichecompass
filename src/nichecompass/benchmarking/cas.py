"""
This module contains the Cell Type Affinity Similiarity (CAS) benchmark for
testing how accurately the latent nearest neighbor graph preserves
cell-type-pair edges from the spatial (physical) nearest neighbor graph.
It is a measure for global cell type neighborhood preservation.
"""

import math
from typing import Optional

import numpy as np
import scipy.sparse as sp
import squidpy as sq
from anndata import AnnData

from .utils import compute_knn_graph_connectivities_and_distances


def compute_cas(
        adata: AnnData,
        cell_type_key: str="cell_type",
        batch_key: Optional[str]=None,
        spatial_knng_key: str="spatial_knng",
        latent_knng_key: str="nichecompass_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="nichecompass_latent",
        n_neighbors: Optional[int]=15,
        n_perms: int=1000,
        n_jobs: int=1,
        seed: int=0) -> float:
    """
    Compute the Cell Type Affinity Similarity (CAS). The CAS measures how
    accurately the latent nearest neighbor graph preserves cell-type-pair edges
    from the spatial (ground truth) nearest neighbor graph. A value of '1'
    indicates perfect cell-type-pair similarity and a value of '0' indicates no
    cell-type-pair similarity at all. The CAS is a variation of the Cell Type
    Affinity Distance which was first introduced by Lohoff, T. et al.
    Integration of spatial and single-cell transcriptomic data elucidates mouse
    organogenesis. Nat. Biotechnol. 40, 74–85 (2022).
    
    If a ´batch_key´ is provided, separate spatial nearest neighbor graphs per
    batch will be computed and are then combined as disconnected components by
    padding with 0s.
    
    If existent, uses precomputed nearest neighbor graphs stored in
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´.
    Alternatively, computes them on the fly using ´spatial_key´, ´latent_key´
    and ´n_neighbors´, , and stores them in
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´ respectively.
    
    Note that the used neighborhood enrichment implementation from squidpy
    slightly deviates from the original method and we construct nearest neighbor
    graphs using the original spatial coordinates and the latent representation
    from a model respectively to compute the similarity. The cell type affinity
    matrices, also called cell-cell contact (ccc) maps are stored in the AnnData
    object.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in
        ´adata.obs[cell_type_key]´, precomputed nearest neighbor graphs stored
        in ´adata.obsp[spatial_knng_key + '_connectivities']´ and
        ´adata.obsp[latent_knng_key + '_connectivities']´ or spatial coordinates
        stored in ´adata.obsm[spatial_key]´ and the latent representation from a
        model stored in ´adata.obsm[latent_key]´.
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    batch_key:
        Key under which the batches are stored in ´adata.obs´. If ´None´, the
        adata is assumed to only have one unique batch.
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
    n_perms:
        Number of permutations used for the neighborhood enrichment score
        calculation.
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    cas:
        Matrix similarity between the latent cell type affinity matrix and the
        spatial (ground truth) cell type affinity matrix (or
        batch-aggregated spatial cell type affinity matrices) as measured by
        one minus the size-normalied Frobenius norm of the element-wise matrix
        differences.
    """
    # Adding '_connectivities' as expected / added by 
    # 'compute_knn_graph_connectivities_and_distances'
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_knng_connectivities_key in adata.obsp:
        print("Using precomputed spatial nearest neighbor graph...")
    elif batch_key is None:
        print("Computing spatial nearest neighbor graph for entire dataset...")
        # 'compute_knn_graph_connectivities_and_distances' returns weighted
        # symmetric knn graph but nhood_enrichment will ignore the weights and
        # treat it as an unweighted knn graph)
        compute_knn_graph_connectivities_and_distances(
                adata=adata,
                feature_key=spatial_key,
                knng_key=spatial_knng_key,
                n_neighbors=n_neighbors,
                random_state=seed,
                n_jobs=n_jobs)
    elif batch_key is not None:
        # Compute spatial nearest neighbor graph for each batch separately,
        # then combine them as disconnected components by padding with 0s
        adata_batch_list = []
        unique_batches = adata.obs[batch_key].unique().tolist()
        for batch in unique_batches:
            print("Computing spatial nearest neighbor graph for "
                  f"{batch_key} {batch}...")
            adata_batch = adata[adata.obs[batch_key] == batch]

            # 'compute_knn_graph_connectivities_and_distances' returns weighted
            # symmetric knn graph but nhood_enrichment will ignore the weights and
            # treat it as an unweighted knn graph)
            compute_knn_graph_connectivities_and_distances(
                    adata=adata_batch,
                    feature_key=spatial_key,
                    knng_key=spatial_knng_key,
                    n_neighbors=n_neighbors,
                    random_state=seed,
                    n_jobs=n_jobs)
            adata_batch_list.append(adata_batch)

        print("Combining spatial nearest neighbor graphs...")
        batch_connectivities = []
        len_before_batch = 0
        for i in range(len(adata_batch_list)):
            if i == 0: # first batch
                after_batch_connectivities_extension = sp.csr_matrix(
                    (adata_batch_list[0].shape[0],
                    (adata.shape[0] -
                    adata_batch_list[0].shape[0])))
                batch_connectivities.append(sp.hstack(
                    (adata_batch_list[0].obsp[
                        spatial_knng_connectivities_key],
                    after_batch_connectivities_extension)))
            elif i == (len(adata_batch_list) - 1): # last batch
                before_batch_connectivities_extension = sp.csr_matrix(
                    (adata_batch_list[i].shape[0],
                    (adata.shape[0] -
                    adata_batch_list[i].shape[0])))
                batch_connectivities.append(sp.hstack(
                    (before_batch_connectivities_extension,
                    adata_batch_list[i].obsp[
                        spatial_knng_connectivities_key])))
            else: # middle batches
                before_batch_connectivities_extension = sp.csr_matrix(
                    (adata_batch_list[i].shape[0], len_before_batch))
                after_batch_connectivities_extension = sp.csr_matrix(
                    (adata_batch_list[i].shape[0],
                    (adata.shape[0] -
                    adata_batch_list[i].shape[0] -
                    len_before_batch)))
                batch_connectivities.append(sp.hstack(
                    (before_batch_connectivities_extension,
                    adata_batch_list[i].obsp[
                        spatial_knng_connectivities_key],
                    after_batch_connectivities_extension)))
            len_before_batch += adata_batch_list[i].shape[0]
        connectivities = sp.vstack(batch_connectivities)
        adata.obsp[spatial_knng_connectivities_key] = connectivities
        adata.uns[f"{spatial_knng_key}_n_neighbors"] = adata_batch_list[0].uns[
            f"{spatial_knng_key}_n_neighbors"]

    print("Computing spatial neighborhood enrichment scores...")
    # Compute cell type affinity matrix for spatial nearest neighbor graph
    sq.gr.nhood_enrichment(adata=adata,
                           cluster_key=cell_type_key,
                           connectivity_key=spatial_knng_key,
                           n_perms=n_perms,
                           seed=seed,
                           show_progress_bar=False)
    
    # Save results in adata (no ´key_added´ functionality in squidpy)
    adata.uns[f"{cell_type_key}_spatial_nhood_enrichment"] = {}
    adata.uns[f"{cell_type_key}_spatial_nhood_enrichment"]["zscore"] = (
        adata.uns[f"{cell_type_key}_nhood_enrichment"]["zscore"])
    del(adata.uns[f"{cell_type_key}_nhood_enrichment"]["zscore"])

    if latent_knng_connectivities_key in adata.obsp:
        print("Using precomputed latent nearest neighbor graph...")
    else:
        print("Computing latent nearest neighbor graph...")
        # 'compute_knn_graph_connectivities_and_distances' returns weighted
        # symmetric knn graph but nhood_enrichment will ignore the weights and
        # treat it as an unweighted knn graph)
        compute_knn_graph_connectivities_and_distances(
                adata=adata,
                feature_key=latent_key,
                knng_key=latent_knng_key,
                n_neighbors=n_neighbors,
                random_state=seed,
                n_jobs=n_jobs)

    print("Computing latent neighborhood enrichment scores...")
    # Compute cell type affinity matrix for latent nearest neighbor graph
    sq.gr.nhood_enrichment(adata,
                           cluster_key=cell_type_key,
                           connectivity_key=latent_knng_key,
                           n_perms=n_perms,
                           seed=seed,
                           show_progress_bar=False)

    # Save results in adata (no ´key_added´ functionality in squidpy)
    adata.uns[f"{cell_type_key}_latent_nhood_enrichment"] = {}
    adata.uns[f"{cell_type_key}_latent_nhood_enrichment"]["zscore"] = (
        adata.uns[f"{cell_type_key}_nhood_enrichment"]["zscore"])
    del adata.uns[f"{cell_type_key}_nhood_enrichment"]["zscore"]

    print("Computing CAS...")
    # Calculate Frobenius norm of the element-wise matrix differences to
    # quantify distance
    nhood_enrichment_zscores_diff = (
        adata.uns[f"{cell_type_key}_latent_nhood_enrichment"]["zscore"] -
        adata.uns[f"{cell_type_key}_spatial_nhood_enrichment"]["zscore"])

    # Remove np.nan ´z_scores´ which can happen as a result of
    # ´sq.gr.nhood_enrichment´ permutation if std is 0
    nhood_enrichment_zscores_diff = (
        nhood_enrichment_zscores_diff[~np.isnan(nhood_enrichment_zscores_diff)])
    
    cad = np.linalg.norm(nhood_enrichment_zscores_diff)

    # Normalize CAD to be between 0 and 1 and convert to CAS by subtracting
    # from 1. First, normalize by the number of cell types, then apply scaling
    # function
    cad_norm = (math.atan(cad / nhood_enrichment_zscores_diff.shape[0]) /
                (math.pi / 2))
    cas = 1 - cad_norm
    return cas


def compute_avg_cas(
        adata: AnnData,
        cell_type_key: str="cell_type",
        spatial_key: str="spatial",
        latent_key: str="nichecompass_latent",
        min_n_neighbors: int=1,
        max_n_neighbors: int=15,
        seed: int=0) -> float:
    """
    Compute multiple Cell Type Affinity Similarities (CAS) by varying the
    number of neighbors used for nearest neighbor graph construction (between
    ´min_n_neighbors´ and ´max_n_neighbors´) and return the average CAS. Can
    use precomputed spatial and latent nearest neighbor graphs stored in
    ´adata.obsp[f'nichecompass_spatial_{n_neighbors}nng_connectivities']´ and
    ´adata.obsp[f'nichecompass_latent_{n_neighbors}nng_connectivities']´
    respectively.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in
        ´adata.obs[cell_type_key]´, spatial coordinates stored in
        ´adata.obsm[spatial_key]´ and the latent representation from a model
        stored in ´adata.obsm[latent_key]´. Precomputed nearest neighbor graphs
        can optionally be stored in
        ´adata.obsp[f'nichecompass_spatial_{n_neighbors}nng_connectivities']´ and
        ´adata.obsp[f'nichecompass_latent_{n_neighbors}nng_connectivities']´.
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from a model is stored in
        ´adata.obsm´.
    min_n_neighbors:
        Minimum number of neighbors used for computing the average CAS.
    max_n_neighbors:
        Maximum number of neighbors used for computing the average CAS.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    avg_cas:
        Average CAS computed over different nearest neighbor graphs with
        varying number of neighbors.
    """
    cas_list = []
    for n_neighbors in range(min_n_neighbors, max_n_neighbors):
        cas_list.append(compute_cas(
            adata=adata,
            cell_type_key=cell_type_key,
            spatial_knng_key=f"{spatial_key}_{n_neighbors}nng",
            latent_knng_key=f"{latent_key}_{n_neighbors}nng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed))
    avg_cas = np.mean(cas_list)
    return avg_cas