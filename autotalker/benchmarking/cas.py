"""
This module contains the Cell Type Affinity Similiarity (CAS) benchmark for
testing how accurately the latent nearest neighbor graph preserves
cell-type-pair edges from the physical (spatial) nearest neighbor graph, a
measure for global cell type neighborhood preservation.
"""

import math
from typing import Optional

import numpy as np
import scanpy as sc
import squidpy as sq
from anndata import AnnData


def compute_avg_cas(
        adata: AnnData,
        cell_type_key: str="cell_type",
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        min_n_neighbors: int=1,
        max_n_neighbors: int=15,
        seed: int=0,
        visualize_ccc_maps: bool=False) -> float:
    """
    Compute multiple Cell Type Affinity Similarities (CAS) by varying the
    number of neighbors used for nearest neighbor graph construction (between
    ´min_n_neighbors´ and ´max_n_neighbors´) and return the average CAS. Can
    use precomputed spatial and latent nearest neighbor graphs stored in
    ´adata.obsp[f'autotalker_spatial_{n_neighbors}nng_connectivities']´ and
    ´adata.obsp[f'autotalker_latent_{n_neighbors}nng_connectivities']´
    respectively.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in
        ´adata.obs[cell_type_key]´, spatial coordinates stored in
        ´adata.obsm[spatial_key]´ and the latent representation from a model
        stored in ´adata.obsm[latent_key]´. Precomputed nearest neighbor graphs
        can optionally be stored in
        ´adata.obsp[f'autotalker_spatial_{n_neighbors}nng_connectivities']´ and
        ´adata.obsp[f'autotalker_latent_{n_neighbors}nng_connectivities']´.
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
    visualize_ccc_maps:
        If ´True´, also visualize the spatial and latent cell type affinity
        matrices (cell-cell-contact maps).

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
            spatial_knng_key=f"autotalker_spatial_{n_neighbors}nng",
            latent_knng_key=f"autotalker_latent_{n_neighbors}nng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed,
            visualize_ccc_maps=visualize_ccc_maps))
    avg_cas = np.mean(cas_list)
    return avg_cas


def compute_cas(
        adata: AnnData,
        cell_type_key: str="cell_type",
        condition_key: Optional[str]=None,
        spatial_knng_key: str="autotalker_spatial_knng",
        latent_knng_key: str="autotalker_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="autotalker_latent",
        n_neighbors: Optional[int]=15,
        seed: int=0,
        visualize_ccc_maps: bool=False) -> float:
    """
    Compute the Cell Type Affinity Similarity (CAS) between the latent nearest
    neighbor graph and the spatial nearest neighbor graph. The CAS measures how
    accurately the latent nearest neighbor graph preserves cell-type-pair edges
    from the spatial (ground truth) nearest neighbor graph. A value of '1'
    indicates perfect cell-type-pair similarity and a value of '0' indicates no
    cell-type-pair similarity at all. The CAS is a variation of the Cell Type
    Affinity Distance which was first introduced by Lohoff, T. et al.
    Integration of spatial and single-cell transcriptomic data elucidates mouse
    organogenesis. Nat. Biotechnol. 40, 74–85 (2022).
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
    matrices, also called cell-cell contact (ccc) maps are stored in the
    AnnData object and can optionally be visualized.

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
    visualize_ccc_maps:
        If ´True´, also visualize the spatial and latent cell type affinity
        matrices (cell-cell-contact maps).

    Returns
    ----------
    cas:
        Matrix similarity between the latent cell type affinity matrix and the
        spatial (ground truth) cell type affinity matrix as measured by one
        minus the size-normalied Frobenius norm of the element-wise matrix
        differences.
    """
    # Adding '_connectivities' as automatically added by sc.pp.neighbors
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_knng_connectivities_key not in adata.obsp:
        if condition_key is not None:
            unique_conditions = adata.obs[condition_key].unique().tolist()
            for condition in unique_conditions:
                adata_condition = adata[adata.obs[condition_key] == condition]
                # TO DO: Make it work for integrated data #
        else:
            sc.pp.neighbors(adata=adata,
                            use_rep=spatial_key,
                            n_neighbors=n_neighbors,
                            random_state=seed,
                            key_added=spatial_knng_key)
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

    # Compute cell type affinity matrix for spatial nearest neighbor graph
    sq.gr.nhood_enrichment(adata,
                           cluster_key=cell_type_key,
                           connectivity_key=spatial_knng_key,
                           n_perms=1000,
                           seed=seed,
                           show_progress_bar=False)

    # Save results in adata (no ´key_added´ functionality in squidpy)
    adata.uns[f"{cell_type_key}_spatial_nhood_enrichment"] = {}
    adata.uns[f"{cell_type_key}_spatial_nhood_enrichment"]["zscore"] = (
        adata.uns[f"{cell_type_key}_nhood_enrichment"]["zscore"])

    if visualize_ccc_maps:
        sq.pl.nhood_enrichment(adata=adata,
                               cluster_key=cell_type_key,
                               mode="zscore",
                               title="Spatial Cell-type Affinity Matrix",
                               figsize=(5, 5))

    # Compute cell type affinity matrix for latent nearest neighbor graph
    sq.gr.nhood_enrichment(adata,
                           cluster_key=cell_type_key,
                           connectivity_key=latent_knng_key,
                           n_perms=1000,
                           seed=seed,
                           show_progress_bar=False)

    # Save results in adata (no ´key_added´ functionality in squidpy)
    adata.uns[f"{cell_type_key}_latent_nhood_enrichment"] = {}
    adata.uns[f"{cell_type_key}_latent_nhood_enrichment"]["zscore"] = (
        adata.uns[f"{cell_type_key}_nhood_enrichment"]["zscore"])

    if visualize_ccc_maps:
        sq.pl.nhood_enrichment(adata=adata,
                               cluster_key=cell_type_key,
                               mode="zscore",
                               title="Latent Cell-type Affinity Matrix",
                               figsize=(5, 5))

    del adata.uns[f"{cell_type_key}_nhood_enrichment"]["zscore"]

    # Calculate Frobenius norm of the element-wise matrix differences to
    # quantify distance
    nhood_enrichment_zscores_diff = (
        adata.uns[f"{cell_type_key}_latent_nhood_enrichment"]["zscore"] -
        adata.uns[f"{cell_type_key}_spatial_nhood_enrichment"]["zscore"])

    # Remove np.nan ´z_scores´ which can happen as a result of
    # ´sq.pl.nhood_enrichment´ permutation
    nhood_enrichment_zscores_diff = (
        nhood_enrichment_zscores_diff[~np.isnan(nhood_enrichment_zscores_diff)])
    
    cad = np.linalg.norm(nhood_enrichment_zscores_diff)

    # Normalize CAD to be between 0 and 1 and convert to CAS by subtracting
    # from 1
    cad_norm = (math.atan(cad / nhood_enrichment_zscores_diff.shape[0]) /
                (math.pi / 2))
    cas = 1 - cad_norm
    return cas
