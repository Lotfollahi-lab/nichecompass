from typing import Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

from .utils import _compute_graph_indices_and_distances, _convert_to_one_hot


def compute_cell_level_log_clisi_ratios(
        adata: AnnData,
        spatial_key: str="spatial",
        latent_key: str="latent_autotalker_fc_gps",
        cell_type_key: str="celltype_mapped_refined",
        knn_graph_n_neighbors: int=15) -> pd.DataFrame:
    """
    First compute the cell-level Cell-type Local Inverse Simpson's Index (CLISI)
    from the spatial coordinates (ground truth) and from the latent 
    representation of the model (latent) respectively. Then divide the latent 
    CLISI by the ground truth CLISI and compute the log2 to get a relative local
    cell-type heterogeneity score as proposed in Heidari, E. et al. Supervised
    spatial inference of dissociated single-cell data with SageNet. bioRxiv 
    2022.04.14.488419 (2022) doi:10.1101/2022.04.14.488419. 
    Log CLISI ratios closer to 0 indicate a latent representation that more 
    accurately preserves the spatial cell-type heterogeneity of the ground truth.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in 
        ´adata.obs[cell_type_key]´, spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in adata.obsm[latent_rep_key].
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    knn_graph_n_neighbors:
        Number of neighbors used for the construction of the knn graph.

    Returns
    ----------
    cell_level_log_clisi_ratios_df:
        Pandas DataFrame that contains the cell-level log CLISI ratios, the 
        distribution of which indicates how well the latent space preserves 
        ground truth spatial cell-type heterogeneity.
    """
    ground_truth_cell_level_clisi = _compute_cell_level_clisi_from_adata(
        adata=adata,
        knn_graph_feature_key=spatial_key,
        knn_graph_n_neighbors=knn_graph_n_neighbors,
        cell_type_key=cell_type_key)

    latent_cell_level_clisi = _compute_cell_level_clisi_from_adata(
        adata=adata,
        knn_graph_feature_key=latent_key,
        knn_graph_n_neighbors=knn_graph_n_neighbors,
        cell_type_key=cell_type_key)

    cell_level_clisi_ratios = latent_cell_level_clisi / ground_truth_cell_level_clisi
    cell_level_log_clisi_ratios = np.log2(cell_level_clisi_ratios)

    cell_level_log_clisi_ratios_df = pd.DataFrame(
        data=cell_level_log_clisi_ratios,
        index=np.arange(0, len(ground_truth_cell_level_clisi)),
        columns=["log_clisi_ratio"])

    return cell_level_log_clisi_ratios_df


def _compute_cell_level_clisi_from_adata(
        adata: AnnData,
        knn_graph_feature_key: str,
        knn_graph_n_neighbors: int,
        cell_type_key: str,
        perplexity: Optional[float]=None) -> np.ndarray:
    """
    Compute the cell-level Cell-type Local Inverse Simpson's Index (CLISI) by
    constructing a k-nearest-neighbors (knn) graph based on features stored in 
    an AnnData object. The cell-level CLISI captures the degree of cell mixing 
    in a local neighborhood around a given cell. The CLISI score indicates the
    effective number of different categories represented in the local
    neighborhood of each cell. If the cells are well-mixed, we might expect the
    CLISI score to be close to the number of unique cell types (e.g. neigborhoods 
    with an equal number of cells from 2 cell types get a cliSI of 2). Note,
    however, that even under perfect mixing, the value would be smaller than the
    number of unique cell types if the absolute number of cells is different for 
    different cell types.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in 
        ´adata.obs[cell_type_key]´ and features for distance calculation stored
        in ´adata.obsm[knn_graph_feature_key]´.
    knn_graph_feature_key:
        Key under which the features for distance calculation are stored in 
        ´adata.obsm´.
    knn_graph_n_neighbors:
        Number of neighbors used for the construction of the knn graph.
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    perplexity:
        Perplexity used for Simpson's Index calculation. By default, perplexity
        is chosen as 1/3 * n_neighbors used in the knn graph.

    Returns
    ----------
    cell_level_clisi:
        1-D NumPy array that contains the cell-level CLISIs.
    """
    knn_indices, knn_distances = _compute_graph_indices_and_distances(
        adata=adata,
        feature_key=knn_graph_feature_key,
        n_neighbors=knn_graph_n_neighbors,
        mode="knn")

    if perplexity is None:
        # Use LISI default perplexity
        perplexity = np.floor(knn_indices.shape[1] / 3)

    cell_type_labels = adata.obs[cell_type_key].cat.codes.values
    n_cell_types = len(np.unique(adata.obs[cell_type_key]))

    cell_level_clisi = _compute_cell_level_clisi_from_knn_graph(
        knn_distances=knn_distances,
        knn_indices=knn_indices,
        cell_type_labels=cell_type_labels,
        n_cell_types=n_cell_types,
        perplexity=perplexity)

    return cell_level_clisi

    
def _compute_cell_level_clisi_from_knn_graph(
        knn_distances: np.ndarray,
        knn_indices: np.ndarray,
        cell_type_labels: np.ndarray,
        n_cell_types: int,
        perplexity: float=15, 
        tolerance: float=1e-5) -> np.ndarray:
    """
    Compute cell-level Cell-Type Local Inverse Simpson's Index (CLISI) from a 
    knn graph. Local Inverse Simpson's Index was first proposed in Korsunsky, I.
    et al. Fast, sensitive and accurate integration of single-cell data with 
    Harmony. Nat. Methods 16, 1289–1296 (2019). The Inverse Simpson's Index is 
    the expected number of cells needed to be sampled before two are drawm from 
    the same category. Thus, this index reports the effective number of 
    categories in a local neighborhood. LISI combines perplexity-based 
    neighborhood construction with the Inverse Simpson's Index to account for 
    distances between neighbors. Adapted from 
    https://github.com/theislab/scib/blob/29f79d0135f33426481f9ff05dd1ae55c8787142/scib/metrics/lisi.py#L310.

    Parameters
    ----------
    knn_distances: 
        2-D NumPy array that contains the distances to the k-nearest-neighbors 
        for each observation/cell (dimensionality: n_cells x n_neighbors).
    knn_indices:
        2-D NumPy array that contains the indices of the k-nearest-neighbors for
        each observation/cell (dimensionality: n_cells x n_neighbors).
    cell_type_labels:
        1-D NumPy array that contains the encoded cell type labels for all cells.
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
    cell_level_clisi:
        1-D NumPy array that contains the cell-level CLISIs.
    """
    n_cells = knn_distances.shape[0]
    # Initialize Gaussian similarity kernel neighbor probabilities
    P = np.zeros(knn_distances.shape[1])
    # Initialize cell-level Cell-Type Local Simpson's Index (CLSI) values
    cell_level_clsi = np.zeros(n_cells)
    log_perplexity = np.log(perplexity)

    # Loop over all cells to compute cell-level CLSI
    for i in range(n_cells):
        ## 1) Get neighbor probabilities using a Gaussian similarity kernel  
        beta = 1 # starting precision of Gaussian similarity kernel
        beta_min = -np.inf
        beta_max = np.inf

        # Get k-nearest-neighbors of current cell
        cell_knn_distances = knn_distances[i, :]

        # Compute the Gaussian similarity kernel neighbor probabilities and 
        # perplexity for the current precision ´beta´
        H, P = hbeta(cell_knn_distances, beta)

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
            H, P = hbeta(cell_knn_distances, beta)
            H_diff = H - log_perplexity
            n_tries += 1

        if H == 0:
            cell_level_clsi[i] = -1 # default value
            continue

        ## 2) Compute cell-level CLSI values
        cell_knn_indices = knn_indices[i, :]
        neighbor_cell_type_labels = cell_type_labels[cell_knn_indices]
        neighbor_cell_types_one_hot = _convert_to_one_hot(
            neighbor_cell_type_labels, n_cell_types)
        # Sum P per cell type
        P_cell_type_sum = np.matmul(P, neighbor_cell_types_one_hot)
        # Sum square of cell-type specific P_sum's
        cell_level_clsi[i] = np.dot(P_cell_type_sum, P_cell_type_sum)

    ## 3) Compute CLISI by inversing CLSI
    cell_level_clisi = 1 / cell_level_clsi

    return cell_level_clisi


def hbeta(D: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a vector of Euclidean distances and the precision of a Gaussian 
    similarity kernel, compute the normalized Gaussian kernel similarity values 
    (neighbor probabilities) as well as the entropy. Adapted from
    https://github.com/theislab/scib/blob/main/scib/metrics/lisi.py#L483.

    Parameters
    ----------
    D:
        Vector of euclidean distances.
    beta:
        Precision of the Gaussian similarity kernel.

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