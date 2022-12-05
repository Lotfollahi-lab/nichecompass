"""
This module contains a benchmark for testing how good the latent neighbor graph
preserves cell type neighborhoods from the original spatial neighbor graph.
"""

import numpy as np
import squidpy as sq
from anndata import AnnData

from autotalker.utils import compute_graph_connectivities


def compute_avg_cad(
        adata: AnnData,
        cell_type_key: str="cell-type",
        spatial_key: str="spatial",
        latent_key: str="latent_autotalker",
        seed: int=0,
        visualize_ccc_maps: bool=False) -> float:
    """
    Compute multiple cell-type affinity distances by varying the number of 
    neighbors used for nearest neighbor graph construction (between 1 and 15)
    and return the average cad.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in 
        ´adata.obs[cell_type_key]´, spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in adata.obsm[latent_key].
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    seed:
        Random seed to get reproducible results.
    visualize_ccc_maps:
        If ´True´, also visualize the spatial/physical and latent cell-type
        affinity matrices (cell-cell contact maps).

    Returns
    ----------
    avg_cad:
        Average cell-type affinity distance computed over different nearest 
        neighbor graphs with varying number of neighbors.
    """
    cad_list = []
    for n_neighbors in range(1,15):
        cad_list.append(_compute_cad(
            adata=adata,
            cell_type_key=cell_type_key,
            spatial_key=spatial_key,
            latent_key=latent_key,
            n_neighbors=n_neighbors,
            seed=seed,
            visualize_ccc_maps=visualize_ccc_maps))
    avg_cad = np.mean(cad_list)
    return avg_cad


def _compute_cad(
        adata: AnnData,
        cell_type_key: str="cell-type",
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        n_neighbors: int=15,
        seed: int=0,
        visualize_ccc_maps: bool=False) -> float:
    """
    Compute the cell-type affinity distance (CAD) as first introduced by Lohoff,
    T. et al. Integration of spatial and single-cell transcriptomic data 
    elucidates mouse organogenesis. Nat. Biotechnol. 40, 74–85 (2022). 
    Note that the used neighborhood enrichment implementation from squidpy 
    slightly deviates from the original method and we construct nearest neighbor
    graphs using the original spatial coordinates and the latent representation
    from the model respectively. The cell-cell contact maps are stored in the
    AnnData object and can optionally be visualized.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in 
        ´adata.obs[cell_type_key]´, spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in adata.obsm[latent_key].
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
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
        Random seed to get reproducible results.
    visualize_ccc_maps:
        If ´True´, also visualize the spatial/physical and latent cell-type
        affinity matrices (cell-cell contact maps).

    Returns
    ----------
    cell_type_affinity_distance:
        Matrix distance between the spatial coordinate (ground truth) cell-type
        affinity matrix and the latent representation cell-type affinity matrix
        as measured by the Frobenius norm of the element-wise matrix 
        differences.
    """
    # Compute physical (ground truth) connectivities
    adata.obsp["cad_spatial_connectivities"] = compute_graph_connectivities(
        adata=adata,
        feature_key=spatial_key,
        n_neighbors=n_neighbors,
        mode="knn",
        seed=seed)

    # Compute latent connectivities
    adata.obsp["cad_latent_connectivities"] = compute_graph_connectivities(
        adata=adata,
        feature_key=latent_key,
        n_neighbors=n_neighbors,
        mode="knn",
        seed=seed)

    # Calculate cell-type affinity scores for spatial nearest neighbor graph
    sq.gr.nhood_enrichment(adata,
                           cluster_key=cell_type_key,
                           connectivity_key="cad_spatial",
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
                               title="Spatial Cell-type Affinity",
                               figsize=(5, 5))

    # Calculate cell type affinity scores for latent nearest neighbor graph
    sq.gr.nhood_enrichment(adata,
                           cluster_key=cell_type_key,
                           connectivity_key="cad_latent",
                           n_perms=1000,
                           seed=seed,
                           show_progress_bar=False)

    # Save results in adata (no ´key_added´ functionality in squidpy)
    adata.uns[f"{cell_type_key}_latent_nhood_enrichment"] = {}
    adata.uns[f"{cell_type_key}_latent_nhood_enrichment"]["zscore"] = adata.uns[
        f"{cell_type_key}_nhood_enrichment"]["zscore"]

    if visualize_ccc_maps:
        sq.pl.nhood_enrichment(adata=adata,
                               cluster_key=cell_type_key,
                               mode="zscore",
                               title="Latent Cell-type Affinity",
                               figsize=(5, 5))

    del adata.uns[f"{cell_type_key}_nhood_enrichment"]["zscore"]

    # Calculate Frobenius norm of matrix differences to quantify distance
    nhood_enrichment_zscores_diff = (
        adata.uns[f"{cell_type_key}_latent_nhood_enrichment"]["zscore"] -
        adata.uns[f"{cell_type_key}_spatial_nhood_enrichment"]["zscore"])
    cell_type_affinity_distance = np.linalg.norm(nhood_enrichment_zscores_diff,
                                                 ord="fro")
    return cell_type_affinity_distance

    
