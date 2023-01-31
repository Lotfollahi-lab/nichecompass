"""
This module contains the Median Absolute Log Relative Cell-Type Local Inverse
Simpson's Index (RCLISI) benchmark for testing how accurately the latent
feature space preserves neighborhood cell-type heterogeneity from the original
spatial feature space.
"""

import numpy as np
import scanpy as sc
from anndata import AnnData

from scib.metrics.lisi import lisi_graph_py

def compute_rclisi(
        adata: AnnData,
        cell_type_key: str="cell-type",
        spatial_knng_key: str="autotalker_spatial_nng",
        latent_knng_key: str="autotalker_latent_nng",
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        n_neighbors: int=15,
        seed: int=0) -> float:
    """
    Compute the Median Absolute Log RCLISI (RCLISI) across all cells. A lower
    value indicates a latent representation that more accurately preserves the
    spatial neighborhood cell-type heterogeneity of the ground truth.

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

    Returns
    ----------
    rclisi:
        The Median Absolute Log RCLISI computed over all cells.
    """
    # Adding '_connectivities' as required by squidpy
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_knng_connectivities_key not in adata.obsp:
        # Compute spatial (ground truth) connectivities
        print("Computing spatial nearest neighbor graph...")
        sc.pp.neighbors(adata=adata,
                        use_rep=spatial_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=spatial_knng_key)
    else:
        print("Using precomputed spatial nearest neighbor graph...")

    if latent_knng_connectivities_key not in adata.obsp:
        print("Computing latent nearest neighbor graph...")
        # Compute latent connectivities
        sc.pp.neighbors(adata=adata,
                        use_rep=latent_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=latent_knng_key)
    else:
        print("Using precomputed latent nearest neighbor graph...")

    adata_tmp = adata.copy()
    adata_tmp.obsp["connectivities"] = (
        adata.obsp[spatial_knng_connectivities_key])

    spatial_cell_clisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key=cell_type_key,
        n_neighbors=n_neighbors,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False)

    adata_tmp.obsp["connectivities"] = (
        adata.obsp[latent_knng_connectivities_key])

    latent_cell_clisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key=cell_type_key,
        n_neighbors=n_neighbors,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False)

    cell_rclisi_scores = latent_cell_clisi_scores / spatial_cell_clisi_scores
    cell_log_rclisi_scores = np.log2(cell_rclisi_scores)

    rclisi = np.median(abs(cell_log_rclisi_scores))
    return rclisi