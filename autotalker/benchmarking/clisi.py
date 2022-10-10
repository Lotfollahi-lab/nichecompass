from typing import Optional

import numpy as np
import pandas as pd
import scipy
import squidpy as sq
from anndata import AnnData
from sklearn.metrics import pairwise_distances

from .utils import _compute_knn_graph


def compute_simpson_index(
    D=None, knn_idx=None, cell_type_labels=None, n_cell_types=None, perplexity=15, tol=1e-5
):
    """
    Simpson index of batch labels subset by group.
    :param D: distance matrix ``n_cells x n_nearest_neighbors``
    :param knn_idx: index of ``n_nearest_neighbors`` of each cell
    :param batch_labels: a vector of length n_cells with batch info
    :param n_batches: number of unique batch labels
    :param perplexity: effective neighborhood size
    :param tol: a tolerance for testing effective neighborhood size
    :returns: the simpson index for the neighborhood of each cell
    """
    n = D.shape[0]
    P = np.zeros(D.shape[1])
    simpson = np.zeros(n)
    logU = np.log(perplexity)

    # loop over all cells
    for i in np.arange(0, n, 1):
        beta = 1
        # negative infinity
        betamin = -np.inf
        # positive infinity
        betamax = np.inf
        # get active row of D
        D_act = D[i, :]
        H, P = Hbeta(D_act, beta)
        Hdiff = H - logU
        tries = 0
        # first get neighbor probabilities
        while np.logical_and(np.abs(Hdiff) > tol, tries < 50):
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            H, P = Hbeta(D_act, beta)
            Hdiff = H - logU
            tries += 1

        if H == 0:
            simpson[i] = -1
            continue

            # then compute Simpson's Index
        non_nan_knn = knn_idx[i][np.invert(np.isnan(knn_idx[i]))].astype("int")
        cell_types = cell_type_labels[non_nan_knn]
        # convertToOneHot omits all nan entries.
        # Therefore, we run into errors in np.matmul.
        if len(cell_types) == len(P):
            B = convert_to_one_hot(cell_types, n_cell_types)
            sumP = np.matmul(P, B)  # sum P per batch
            simpson[i] = np.dot(sumP, sumP)  # sum squares
        else:  # assign worst possible score
            simpson[i] = 1

    return simpson


def Hbeta(D_row, beta):
    """
    Helper function for simpson index computation
    """
    P = np.exp(-D_row * beta)
    sumP = np.nansum(P)
    if sumP == 0:
        H = 0
        P = np.zeros(len(D_row))
    else:
        H = np.log(sumP) + beta * np.nansum(D_row * P) / sumP
        P /= sumP
    return H, P


def convert_to_one_hot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output 2-D array of one-hot vectors,
    where an i'th input value of j will set a '1' in the i'th row, j'th column of the
    output array.
    Example:
    .. code-block:: python
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print(one_hot_v)
    .. code-block::
        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    # assert isinstance(vector, np.ndarray)
    # assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector) + 1
    # else:
    #    assert num_classes > 0
    #    assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def _compute_clisi(adata: AnnData,
                   knn_graph_feature_key: str,
                   knn_graph_n_neighbors: int,
                   cell_type_key: str,
                   perplexity: Optional[float]=None):
    """
    Compute cell-type lisi score (clisi) on kNN graph provided in the adata object. 
    
    perplexity:
        Perplexity used for simpson index calculation. By default, perplexity
        is chosen as 1/3 * n_neighbors used in the knn graph.
    # Create neighbor graph from spatial coordinates
    sq.gr.spatial_neighbors(adata,
                            spatial_key=spatial_key,
                            coord_type="generic",
                            n_neighs=neighborhood_graph_n_neighs,
                            key_added="clisi_spatial")


    sparse_dist_mat = scipy.sparse.find(adata.obsp["clisi_spatial_distances"])
    neighbor_index_mat = sparse_dist_mat[0].reshape(sparse_dist_mat[1][-1] + 1,
                                                    neighborhood_graph_n_neighs)
    neighbor_dist_mat = sparse_dist_mat[2].reshape(sparse_dist_mat[1][-1] + 1,
                                                   neighborhood_graph_n_neighs)
    """

    knn_indices, knn_distances = _compute_knn_graph(
        adata=adata,
        feature_key=knn_graph_feature_key,
        n_neighbors=knn_graph_n_neighbors)

    if perplexity is None:
        # Use LISI default perplexity
        perplexity = np.floor(knn_indices.shape[1] / 3)

    cell_type_labels = adata.obs[cell_type_key].cat.codes.values
    n_cell_types = len(np.unique(adata.obs[cell_type_key]))

    clsi = compute_simpson_index(
        D=knn_distances,
        knn_idx=knn_indices,
        cell_type_labels=cell_type_labels,
        n_cell_types=n_cell_types,
        perplexity=perplexity)

    clisi = 1 / clsi
    return clisi


def compute_cell_level_log_clisi_ratios(
        adata: AnnData,
        spatial_key: str="spatial",
        latent_key: str="latent_autotalker_fc_gps",
        cell_type_key: str="celltype_mapped_refined",
        knn_graph_n_neighbors: int=6):
    """
    First compute the cell-level Cell-type Local Inverse Simpson Index (CLISI)
    from the spatial coordinates (ground truth) and from the latent 
    representation of the model (latent) respectively. The cell-level CLISI 
    captures the degree of cell mixing in a local neighborhood around a given
    cell. Then divide the latent CLISI by the ground truth CLISI and compute the
    log2 to get a relative local heterogeneity score as proposed in Heidari, E. 
    et al. Supervised spatial inference of dissociated single-cell data with
    SageNet. bioRxiv 2022.04.14.488419 (2022) doi:10.1101/2022.04.14.488419. 
    Values closer to 0 indicate a latent representation that more accurately 
    preserves the spatial cell-type heterogeneity of the ground truth.

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
        Pandas DataFrame that contains the cell-level CLISI ratios, the 
        distribution of which indicates how well the latent space preserves 
        ground truth spatial cell-type heterogeneity.
    """
    ground_truth_cell_level_clisi = _compute_cell_level_clisi(
        adata=adata,
        knn_graph_feature_key=spatial_key,
        knn_graph_n_neighbors=knn_graph_n_neighbors,
        cell_type_key=cell_type_key)

    latent_cell_level_clisi = _compute_cell_level_clisi(
        adata=adata,
        knn_graph_feature_key=latent_key,
        knn_graph_n_neighbors=knn_graph_n_neighbors,
        cell_type_key=cell_type_key)

    cell_level_clisi_ratios = latent_cell_level_clisi / ground_truth_cell_level_clisi
    cell_level_log_clisi_ratios = np.log2(cell_level_clisi_ratios)

    cell_level_log_clisi_ratios_df = pd.DataFrame(
        data=cell_level_log_clisi_ratios,
        index=np.arange(0, len(ground_truth_cell_level_clisi)))
    
    return cell_level_log_clisi_ratios_df