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
                   perplexity=None):
    """
    Compute LISI score on kNN graph provided in the adata object. By default, perplexity
    is chosen as 1/3 * number of nearest neighbours in the knn-graph.
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

    neighbor_index_mat, neighbor_dist_mat = _compute_knn_graph(adata, distance_key, knn_graph_n_neighbors)

    if perplexity is None:
        # Use LISI default perplexity
        perplexity = np.floor(neighbor_index_mat.shape[1] / 3)

    cell_type_codes = adata.obs[cell_type_key].cat.codes.values
    n_cell_types = len(np.unique(adata.obs[cell_type_key]))

    simpson_estimate_label = compute_simpson_index(
        D=neighbor_dist_mat,
        knn_idx=neighbor_index_mat,
        cell_type_labels=cell_type_codes,
        n_cell_types=n_cell_types,
        perplexity=perplexity,
    )
    simpson_est_label = 1 / simpson_estimate_label
    return simpson_est_label
    # extract results
    d = {cell_type_key: simpson_est_label}
    lisi_estimate = pd.DataFrame(data=d, index=np.arange(0, len(simpson_est_label)))

    return lisi_estimate


def compute_clisi_metric(adata: AnnData,
                         spatial_key: str="spatial",
                         latent_rep_key: str="latent_autotalker_fc_gps",
                         cell_type_key: str="celltype_mapped_refined",
                         neighborhood_graph_n_neighs: int=6):
    """
    
    """
    spatial_clisi = _compute_clisi(
        adata,
        spatial_key=spatial_key,
        cell_type_key=cell_type_key,
        neighborhood_graph_n_neighs=neighborhood_graph_n_neighs)

    latent_clisi = _compute_clisi(
        adata,
        spatial_key=latent_rep_key,
        cell_type_key=cell_type_key,
        neighborhood_graph_n_neighs=neighborhood_graph_n_neighs)

    relative_clisi = latent_clisi / spatial_clisi

    log_relative_clisi = np.log2(relative_clisi)

    lisi_estimate = pd.DataFrame(data=log_relative_clisi,
                                 index=np.arange(0, len(spatial_clisi)))
    
    return lisi_estimate

    

