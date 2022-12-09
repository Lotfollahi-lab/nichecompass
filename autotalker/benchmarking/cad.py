"""
This module contains the Cell-Type Affinity Distance (CAD) benchmark for testing
how good the latent nearest neighbor graph preserves cell-type-pair edges from
the original spatial nearest neighbor graph.
"""

from typing import Optional

import numpy as np
import squidpy as sq
from anndata import AnnData

from autotalker.utils import compute_graph_connectivities


def compute_avg_cad(
        adata: AnnData,
        cell_type_key: str="cell-type",
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        min_n_neighbors: int=1,
        max_n_neighbors: int=16,
        seed: int=0,
        visualize_ccc_maps: bool=False) -> float:
    """
    Compute multiple Cell-Type Affinity Distances (CADs) by varying the number
    of neighbors used for nearest neighbor graph construction (between
    ´min_n_neighbors´ and ´max_n_neighbors´) and return the average CAD. Can use
    precomputed spatial and latent nearest neighbor graphs stored in
    ´adata.obsp[f'autotalker_spatial_{n_neighbors}nng_connectivities']´ and
    ´adata.obsp[f'autotalker_latent_{n_neighbors}nng_connectivities']´
    respectively.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in
        ´adata.obs[cell_type_key]´, spatial coordinates stored in
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in ´adata.obsm[latent_key]´. Precomputed nearest neighbor graphs
        can optionally be stored in
        ´adata.obsp[f'autotalker_spatial_{n_neighbors}nng_connectivities']´ and
        ´adata.obsp[f'autotalker_latent_{n_neighbors}nng_connectivities']´.
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from the model is stored in
        ´adata.obsm´.
    min_n_neighbors:
        Minimum number of neighbors used for computing the average CAD.
    max_n_neighbors:
        Maximum number of neighbors used for computing the average CAD.
    seed:
        Random seed for reproducibility.
    visualize_ccc_maps:
        If ´True´, also visualize the spatial and latent cell-type affinity
        matrices (ccc maps).

    Returns
    ----------
    avg_cad:
        Average CAD computed over different nearest neighbor graphs with varying
        number of neighbors.
    """
    cad_list = []
    for n_neighbors in range(min_n_neighbors, max_n_neighbors):
        cad_list.append(compute_cad(
            adata=adata,
            cell_type_key=cell_type_key,
            spatial_knng_key=f"autotalker_spatial_{n_neighbors}nng",
            latent_knng_key=f"autotalker_latent_{n_neighbors}nng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed,
            visualize_ccc_maps=visualize_ccc_maps))
    avg_cad = np.mean(cad_list)
    return avg_cad


def compute_cad(
        adata: AnnData,
        cell_type_key: str="cell-type",
        spatial_knng_key: str="autotalker_spatial_8nng",
        latent_knng_key: str="autotalker_latent_8nng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="autotalker_latent",
        n_neighbors: Optional[int]=8,
        seed: Optional[int]=0,
        visualize_ccc_maps: bool=False) -> float:
    """
    Compute the Cell-type Affinity Distance (CAD) between the latent nearest
    neighbor graph and the spatial nearest neighbor graph. A lower value
    indicates a latent nearest neighbor graph that more accurately preserves
    cell-type-pair edges from the spatial (ground truth) nearest neighbor
    graph. The CAD was first introduced by Lohoff, T. et al. Integration of
    spatial and single-cell transcriptomic data elucidates mouse organogenesis.
    Nat. Biotechnol. 40, 74–85 (2022).
    If existent, use precomputed nearest neighbor graphs stored in
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´.
    Alternatively, compute them on the fly using ´spatial_key´, ´latent_key´ and
    ´n_neighbors´. Note that the used neighborhood enrichment implementation
    from squidpy slightly deviates from the original method and we construct
    nearest neighbor graphs using the original spatial coordinates and the
    latent representation from the model respectively. The cell-type affinity
    matrices, also called cell-cell contact (ccc) maps are stored in the AnnData
    object and can optionally be visualized.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in
        ´adata.obs[cell_type_key]´, precomputed nearest neighbor graphs stored
        in ´adata.obsp[spatial_knng_key + '_connectivities']´ and
        ´adata.obsp[latent_knng_key + '_connectivities']´ or, alternatively,
        spatial coordinates stored in ´adata.obsm[spatial_key]´ and the latent
        representation from the model stored in ´adata.obsm[latent_key]´.
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
        Key under which the latent representation from the model is stored in
        ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the construction of the nearest neighbor
        graphs from the spatial coordinates and the latent representation from
        the model.
    seed:
        Random seed for reproducibility.
    visualize_ccc_maps:
        If ´True´, also visualize the spatial and latent cell-type affinity
        matrices (ccc maps).

    Returns
    ----------
    cad:
        Matrix distance between the latent cell-type affinity matrix and the
        spatial (ground truth) cell-type affinity matrix as measured by the
        Frobenius norm of the element-wise matrix differences.
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

    # Compute cell-type affinity matrix for spatial nearest neighbor graph
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

    # Compute cell-type affinity matrix for latent nearest neighbor graph
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
    cad = np.linalg.norm(nhood_enrichment_zscores_diff,
                         ord="fro")
    return cad
