"""
This module contains the Cell Type Local Inverse Simpson's Index Similarity
(CLISIS) benchmark for testing how accurately the latent nearest neighbor graph
preserves local neighborhood cell type heterogeneity from the spatial (physical)
nearest neighbor graph. It is a measure for local cell type neighborhood
preservation.
"""

from typing import Optional

import numpy as np
from anndata import AnnData
from scib_metrics import lisi_knn

from .utils import compute_knn_graph_connectivities_and_distances


def compute_clisis(
        adata: AnnData,
        cell_type_key: str="cell_type",
        batch_key: Optional[str]=None,
        spatial_knng_key: str="spatial_knng",
        latent_knng_key: str="nichecompass_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="nichecompass_latent",
        n_neighbors: Optional[int]=90,
        n_jobs: int=1,
        seed: int=0) -> float:
    """
    Compute the Cell Type Local Inverse Simpson's Index Similarity (CLISIS). The
    CLISIS measures how accurately the latent nearest neighbor graph preserves
    local neighborhood cell type heterogeneity from the spatial nearest neighbor
    graph. The CLISIS ranges between '0' and '1' with higher values indicating
    better local neighborhood cell type heterogeneity preservation. It is
    computed by first calculating the Cell Type Local Inverse Simpson's Index
    (CLISI) as proposed by Luecken, M. D. et al. Benchmarking atlas-level data
    integration in single-cell genomics. Nat. Methods 19, 41–50 (2022) on the
    latent and spatial nearest neighbor graph(s) respectively.* Afterwards, the
    ratio of the two CLISI scores is taken and logarithmized as proposed by
    Heidari, E. et al. Supervised spatial inference of dissociated single-cell
    data with SageNet. bioRxiv 2022.04.14.488419 (2022)
    doi:10.1101/2022.04.14.488419, leveraging the properties of the log that
    np.log2(x/y) = -np.log2(y/x) and np.log2(x/x) = 0. At this stage, values
    closer to 0 indicate better local neighborhood cell type heterogeneity
    preservation. We then normalize the resulting value by the maximum possible
    value that would occur in the case of minimal local neighborhood cell type
    preservation to scale our metric between '0' and '1'. Finally, we compute
    the median of the absolute normalized scores and subtract it from 1 so that
    values closer to '1' indicate better local neighborhood cell type
    heterogeneity preservation.
    
    If a ´batch_key´ is provided, separate spatial nearest neighbor graphs per
    batch will be computed and the spatial clisi scores are computed for each
    batch separately.
    
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
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    clisis:
        The Cell Type Local Inverse Simpson's Index Similarity.
    """
    # Adding '_connectivities' as expected / added by 
    # 'compute_knn_graph_connectivities_and_distances'
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_knng_connectivities_key in adata.obsp:
        print("Using precomputed spatial nearest neighbor graph...")

        print("Computing spatial cell CLISI scores for entire dataset...")
        spatial_cell_clisi_scores = lisi_knn(
            X=adata.obsp[f"{spatial_knng_key}_distances"],
            labels=adata.obs[cell_type_key])
        
    elif batch_key is None:
        print("Computing spatial nearest neighbor graph for entire dataset...")  
        compute_knn_graph_connectivities_and_distances(
                adata=adata,
                feature_key=spatial_key,
                knng_key=spatial_knng_key,
                n_neighbors=n_neighbors,
                random_state=seed,
                n_jobs=n_jobs)
        
        print("Computing spatial cell CLISI scores for entire dataset...")
        spatial_cell_clisi_scores = lisi_knn(
            X=adata.obsp[f"{spatial_knng_key}_distances"],
            labels=adata.obs[cell_type_key])

    elif batch_key is not None:
        # Compute cell CLISI scores for spatial nearest neighbor graph
        # of each batch separately and add to array
        unique_batches = adata.obs[batch_key].unique().tolist()
        spatial_cell_clisi_scores = np.zeros(len(adata))
        adata.obs["index"] = np.arange(len(adata))
        for batch in unique_batches:
            adata_batch = adata[adata.obs[batch_key] == batch]
            
            print("Computing spatial nearest neighbor graph for "
                  f"{batch_key} {batch}...")
            # Compute batch-specific spatial (ground truth) nearest
            # neighbor graph
            compute_knn_graph_connectivities_and_distances(
                    adata=adata_batch,
                    feature_key=spatial_key,
                    knng_key=spatial_knng_key,
                    n_neighbors=n_neighbors,
                    random_state=seed,
                    n_jobs=n_jobs)
            
            print("Computing spatial cell CLISI scores for "
                  f"{batch_key} {batch}...")
            batch_spatial_cell_clisi_scores = lisi_knn(
                X=adata_batch.obsp[f"{spatial_knng_key}_distances"],
                labels=adata_batch.obs[cell_type_key])
            
            # Save results
            spatial_cell_clisi_scores[adata_batch.obs["index"].values] = (
                batch_spatial_cell_clisi_scores)  

    if latent_knng_connectivities_key in adata.obsp:
        print("Using precomputed latent nearest neighbor graph...")
    else:
        print("Computing latent nearest neighbor graph...")
        # Compute latent connectivities
        compute_knn_graph_connectivities_and_distances(
                adata=adata,
                feature_key=latent_key,
                knng_key=latent_knng_key,
                n_neighbors=n_neighbors,
                random_state=seed,
                n_jobs=n_jobs)

    print("Computing latent cell CLISI scores...")
    latent_cell_clisi_scores = lisi_knn(
        X=adata.obsp[f"{latent_knng_key}_distances"],
        labels=adata.obs[cell_type_key])

    print("Computing CLISIS...")
    cell_rclisi_scores = latent_cell_clisi_scores / spatial_cell_clisi_scores
    cell_log_rclisi_scores = np.log2(cell_rclisi_scores)

    n_cell_types = adata.obs[cell_type_key].nunique()
    max_cell_log_rclisi = np.log2(n_cell_types / 1)
    norm_cell_log_rclisi_scores = cell_log_rclisi_scores / max_cell_log_rclisi

    clisis = (1 - np.nanmedian(abs(norm_cell_log_rclisi_scores)))
    return clisis