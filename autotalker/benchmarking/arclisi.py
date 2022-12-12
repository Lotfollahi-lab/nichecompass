"""
This module contains the Average Absolute Log Relative Cell-Type Local Inverse
Simpson's Index (ARCLISI) benchmark for testing how accurately the latent
feature space preserves neighborhood cell-type heterogeneity from the original
spatial feature space.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

from autotalker.utils import compute_graph_indices_and_distances
from .utils import convert_to_one_hot


def compute_arclisi(
        adata: AnnData,
        cell_type_key: str="cell-type",
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        n_neighbors: int=8,
        seed: int=0) -> float:
    """
    Compute the Average Absolute Log RCLISI (ARCLISI) across all cells. A lower
    value indicates a latent representation that more accurately preserves the
    spatial neighborhood cell-type heterogeneity of the ground truth.

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
        Random seed for reproducibility.

    Returns
    ----------
    arclisi:
        The Average Absolute Log RCLISI computed over all cells.
    """
    per_cell_log_rclisi_df = compute_per_cell_log_rclisi(
        adata=adata,
        cell_type_key=cell_type_key,
        spatial_key=spatial_key,
        latent_key=latent_key,
        n_neighbors=n_neighbors,
        seed=seed)

    arclisi = abs(per_cell_log_rclisi_df["log_rclisi"]).mean()
    return arclisi


def compute_per_cell_log_rclisi(
        adata: AnnData,
        cell_type_key: str="cell-type",
        spatial_key: str="spatial",
        latent_key: str="autotalker_latent",
        n_neighbors: int=8,
        seed: int=0) -> pd.DataFrame:
    """
    First, compute the per cell Cell-type Local Inverse Simpson's Index (CLISI)
    from the spatial coordinates (ground truth) and from the latent
    representation of the model respectively. Then divide the latent CLISI by
    the ground truth CLISI and compute the log2 to get a relative local
    cell-type heterogeneity score, the log Relative Cell-type Local Inverse
    Simpson's Index (RCLISI) as proposed in Heidari, E. et al. Supervised
    spatial inference of dissociated single-cell data with SageNet. bioRxiv
    2022.04.14.488419 (2022) doi:10.1101/2022.04.14.488419.
    A log RCLISI closer to 0 indicates a latent representation that more
    accurately preserves the spatial cell-type heterogeneity of the ground
    truth.

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
        Random seed for reproducibility.

    Returns
    ----------
    per_cell_log_rclisi_df:
        Pandas DataFrame that contains the per cell log RCLISI, the
        distribution of which indicates how well the latent space preserves
        ground truth spatial cell-type heterogeneity.
    """
    spatial_per_cell_clisi = _compute_per_cell_clisi_from_feature(
        adata=adata,
        cell_type_key=cell_type_key,
        feature_key=spatial_key,
        n_neighbors=n_neighbors,
        seed=seed)

    latent_per_cell_clisi = _compute_per_cell_clisi_from_feature(
        adata=adata,
        cell_type_key=cell_type_key,
        feature_key=latent_key,
        n_neighbors=n_neighbors,
        seed=seed)

    per_cell_rclisi = latent_per_cell_clisi / spatial_per_cell_clisi
    per_cell_log_rclisi = np.log2(per_cell_rclisi)
    per_cell_log_rclisi_df = pd.DataFrame(
        data=per_cell_log_rclisi,
        index=np.arange(0, len(spatial_per_cell_clisi)),
        columns=["log_rclisi"])
    return per_cell_log_rclisi_df


def _compute_per_cell_clisi_from_feature(
        adata: AnnData,
        feature_key: str,
        n_neighbors: int,
        cell_type_key: str,
        perplexity: Optional[float]=None,
        seed: int=0) -> np.ndarray:
    """
    Compute the per cell Cell-type Local Inverse Simpson's Index (CLISI) by
    constructing nearest neighbor graph distances based on
    ´adata.obsm[feature_key]´. The per cell CLISI captures the degree of cell
    mixing in a local neighborhood around a given cell. The CLISI score
    indicates the effective number of different categories represented in the
    local neighborhood of each cell. If the cells are well mixed, we might
    expect the CLISI score to be close to the number of unique cell types (e.g.
    neigborhoods with an equal number of cells from 2 cell types get a CLISI of
    2). Note, however, that even under perfect mixing, the value would be
    smaller than the number of unique cell types if the absolute number of cells
    is different for different cell types.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in
        ´adata.obs[cell_type_key]´ and features for distance calculation stored
        in ´adata.obsm[feature_key]´.
    feature_key:
        Key under which the feature values for distance calculation are stored
        in ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the nearest neighbors graph construction.
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    perplexity:
        Perplexity used for Simpson's Index calculation. By default, perplexity
        is chosen as ´1/3 * n_neighbors´.
    seed:
        Random seed for reproducibility.

    Returns
    ----------
    per_cell_clisi:
        1-D NumPy array that contains the per cell CLISIs.
    """
    knn_indices, knn_distances = compute_graph_indices_and_distances(
        adata=adata,
        feature_key=feature_key,
        n_neighbors=n_neighbors,
        mode="knn",
        seed=seed)

    if perplexity is None:
        if knn_indices.shape[1] < 3:
            perplexity = 1.0
        else:
            # Use LISI default perplexity
            perplexity = np.floor(knn_indices.shape[1] / 3)

    cell_type_labels = adata.obs[cell_type_key].cat.codes.values
    n_cell_types = len(np.unique(adata.obs[cell_type_key]))

    per_cell_clisi = _compute_per_cell_clisi(
        knn_distances=knn_distances,
        knn_indices=knn_indices,
        cell_type_labels=cell_type_labels,
        n_cell_types=n_cell_types,
        perplexity=perplexity)
    return per_cell_clisi

    
def _compute_per_cell_clisi(
        knn_distances: np.ndarray,
        knn_indices: np.ndarray,
        cell_type_labels: np.ndarray,
        n_cell_types: int,
        perplexity: float=15, 
        tolerance: float=1e-5) -> np.ndarray:
    """
    Compute per cell Cell-Type Local Inverse Simpson's Index (CLISI) from
    nearest neighbor graph distances. The LISI was first proposed in Korsunsky,
    I. et al. Fast, sensitive and accurate integration of single-cell data with
    Harmony. Nat. Methods 16, 1289–1296 (2019). The Inverse Simpson's Index is
    the expected number of cells needed to be sampled before two are drawn from
    the same category. Thus, this index reports the effective number of
    categories in a local neighborhood. LISI combines perplexity-based
    neighborhood construction with the Inverse Simpson's Index to account for
    distances between neighbors.
    
    Parts of the implementation are adapted from
    https://github.com/theislab/scib/blob/29f79d0135f33426481f9ff05dd1ae55c8787142/scib/metrics/lisi.py#L310
    (05.12.22).

    Parameters
    ----------
    knn_distances: 
        2-D NumPy array that contains the distances to the k-nearest-neighbors 
        for each observation / cell (dim: (n_cells, n_neighbors)).
    knn_indices:
        2-D NumPy array that contains the indices of the k-nearest-neighbors for
        each observation / cell dim: (n_cells, n_neighbors)).
    cell_type_labels:
        1-D NumPy array that contains the encoded cell type labels for all
        cells.
    n_cell_types:
        Number of unique cell types.
    perplexity:
        Perplexity used for Simpson's Index calculation (effective neighborhood
        size).
    tolerance:
        Tolerance for the effective neighborhood size of the Gaussian similarity
        kernel.

    Returns
    ----------
    per_cell_clisi:
        1-D NumPy array that contains the per cell CLISIs.
    """
    n_cells = knn_distances.shape[0]
    # Initialize Gaussian similarity kernel neighbor probabilities
    P = np.zeros(knn_distances.shape[1])
    # Initialize per cell Cell-Type Local Simpson's Index (CLSI) values
    per_cell_clsi = np.zeros(n_cells)
    log_perplexity = np.log(perplexity)

    # Loop over all cells to compute per cell CLSI
    for i in range(n_cells):
        ## 1) Get neighbor probabilities using a Gaussian similarity kernel
        beta = 1 # starting precision of Gaussian similarity kernel
        beta_min = -np.inf
        beta_max = np.inf

        # Get k-nearest-neighbors of current cell
        cell_knn_distances = knn_distances[i, :]

        # Compute the Gaussian similarity kernel neighbor probabilities and
        # perplexity for the current precision ´beta´
        H, P = _hbeta(cell_knn_distances, beta)

        # Evaluate whether the perplexity difference is within tolerance and if
        # not, increase or decrease precision of the Gaussian similarity kernel
        H_diff = H - log_perplexity
        n_tries = 0
        while np.logical_and(np.abs(H_diff) > tolerance, n_tries < 50):
            if H_diff > 0: # actual perplexity > target perplexity
                # Increase precision (decrease bandwidth)
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                # Decrease precision (increase bandwidth)
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + beta_min) / 2

            # Recompute the Gaussian similarity kernel neighbor probabilities
            # and perplexity with updated precision ´beta´
            H, P = _hbeta(cell_knn_distances, beta)
            H_diff = H - log_perplexity
            n_tries += 1

        if H == 0:
            per_cell_clsi[i] = -1 # default value
            continue

        ## 2) Compute per cell CLSI values
        cell_knn_indices = knn_indices[i, :]
        neighbor_cell_type_labels = cell_type_labels[cell_knn_indices]
        neighbor_cell_types_one_hot = convert_to_one_hot(
            neighbor_cell_type_labels, n_cell_types)
        # Sum P per cell-type
        P_cell_type_sum = np.matmul(P, neighbor_cell_types_one_hot)
        # Sum square of cell-type-specific P_sum's
        per_cell_clsi[i] = np.dot(P_cell_type_sum, P_cell_type_sum)

    ## 3) Compute CLISI by inversing CLSI
    per_cell_clisi = 1 / per_cell_clsi
    return per_cell_clisi


def _hbeta(D: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a vector of Euclidean distances and the precision of a Gaussian
    similarity kernel, compute the normalized Gaussian kernel similarity values
    (neighbor probabilities) as well as the entropy.
    
    Implementation is adapted from
    https://github.com/theislab/scib/blob/main/scib/metrics/lisi.py#L483
    (01.11.2022).

    Parameters
    ----------
    D:
        Vector of euclidean distances.
    beta:
        Precision of the Gaussian similarity kernel.

    Returns
    ----------
    H:
        Gaussian kernel perplexity.
    P:
        Normalized gaussian kernel similarity values (neighbor probabilities).
    """
    P = np.exp(-D * beta)
    P_sum = np.sum(P)
    H = np.log(P_sum) + beta * np.sum(D * P) / P_sum
    P = P / P_sum
    return H, P