"""
This module contains the Cell Type Local Inverse Simpson's Index Similarity
(CLISIS) benchmark for testing how accurately the latent nearest neighbor graph
preserves local neighborhood cell type heterogeneity from the physical (spatial)
nearest neighbor graph.
"""

from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData
from scib.metrics.lisi import lisi_graph_py


def compute_clisis(
        adata: AnnData,
        cell_type_key: str="cell_type",
        spatial_knng_key: str="autotalker_spatial_knng",
        latent_knng_key: str="autotalker_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="autotalker_latent",
        n_neighbors: Optional[int]=15,
        lisi_graph_n_neighbors: int=90,
        seed: int=0) -> float:
    """
    Compute the Cell Type Local Inverse Simpson's Index Similarity (CLISIS). The
    CLISIS measures how accurately the latent nearest neighbor graph preserves
    local neighborhood cell type heterogeneity from the spatial (ground truth)
    nearest neighbor graph. The CLISIS ranges between '0' and '1' with higher
    values indicating better local neighborhood cell type heterogeneity
    preservation. It is computed by first calculating the Cell Type Local
    Inverse Simpson's Index (CLISI) as proposed by Luecken, M. D. et al.
    Benchmarking atlas-level data integration in single-cell genomics. Nat.
    Methods 19, 41–50 (2022) on the latent and spatial nearest neighbor graphs
    respectively.* Afterwards, the ratio of the two CLISI scores is taken and
    logarithmized as proposed by Heidari, E. et al. Supervised spatial inference
    of dissociated single-cell data with SageNet. bioRxiv 2022.04.14.488419
    (2022) doi:10.1101/2022.04.14.488419, leveraging the properties of the log
    that np.log2(x/y) = -np.log2(y/x) and np.log2(x/x) = 0. At this stage,
    values closer to 0 indicate better local neighborhood cell type
    heterogeneity preservation. We then normalize the resulting value by the
    maximum possible value that would occur in the case of minimal local
    neighborhood cell ype preservation to scale our metric between '0' and '1'.
    Finally, we compute the median of the absolute normalized scores and
    subtract it from 1 so that values closer to '1' indicate better local
    neighborhood cell type heterogeneity preservation.
    If existent, uses precomputed nearest neighbor graphs stored in
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´.
    Alternatively, computes them on the fly using ´spatial_key´, ´latent_key´
    and ´n_neighbors´, and stores them in
    ´adata.obsp[spatial_knng_key + '_connectivities']´ and
    ´adata.obsp[latent_knng_key + '_connectivities']´ respectively.    

    * The Inverse Simpson's Index measures the expected number of
    samples needed to be sampled before two are drawn from the same category.
    The Local Inverse Simpson's Index combines perplexity-based neighborhood
    construction with the Inverse Simpson's Index to account for distances
    between neighbors. The CLISI score is the LISI applied to cell nearest
    neighbor graphs with cell types as categories, and indicates the effective
    number of different cell types represented in the local neighborhood of each
    cell. If the cells are well mixed, we might expect the CLISI score to be
    close to the number of unique cell types (e.g. neigborhoods with an equal
    number of cells from 2 cell types get a CLISI of 2). Note, however, that
    even under perfect mixing, the value would be smaller than the number of
    unique cell types if the absolute number of cells is different for different
    cell types.

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
    lisi_graph_n_neighbors:
        Number of neighbors used for the LISI computation.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    clisis:
        The Cell Type Local Inverse Simpson's Index Similarity.
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

    # Create tmp adata as scib does not allow to pass custom keys for
    # connectivities and neighbors
    adata_tmp = adata.copy()
    adata_tmp.obsp["connectivities"] = (
        adata.obsp[spatial_knng_connectivities_key])
    adata_tmp.uns["neighbors"] = (
        adata.uns[spatial_knng_key])

    spatial_cell_clisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key=cell_type_key,
        n_neighbors=lisi_graph_n_neighbors,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False)

    adata_tmp.obsp["connectivities"] = (
        adata.obsp[latent_knng_connectivities_key])
    adata_tmp.uns["neighbors"] = (
        adata.uns[latent_knng_key])

    latent_cell_clisi_scores = lisi_graph_py(
        adata=adata_tmp,
        obs_key=cell_type_key,
        n_neighbors=lisi_graph_n_neighbors,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False)

    cell_rclisi_scores = latent_cell_clisi_scores / spatial_cell_clisi_scores
    cell_log_rclisi_scores = np.log2(cell_rclisi_scores)

    n_cell_types = adata.obs[cell_type_key].nunique()
    max_cell_log_rclisi = np.log2(n_cell_types / 1)
    norm_cell_log_rclisi_scores = cell_log_rclisi_scores / max_cell_log_rclisi

    clisis = np.median(abs(norm_cell_log_rclisi_scores))
    return clisis

