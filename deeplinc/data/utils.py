import numpy as np
import scipy.sparse as sp
# import torch

# from .dataset import SpatialAnnDataDataset


def normalize_adj_mx(adj_mx):
    """
    Symmetrically normalize adjacency matrix as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016). Calculate
    D**(-1/2)*A*D**(-1/2) where D is the degree matrix and A is the adjacency
    matrix where diagonal elements are set to 1, i.e. every node is connected
    to itself.
    """
    adj_mx = sp.coo_matrix(adj_mx)  # convert to sparse matrix COOrdinate format
    adj_mx_ = adj_mx + sp.eye(adj_mx.shape[0])  # add 1s on diagonal
    rowsums = np.array(adj_mx_.sum(1))  # calculate sums over rows
    degree_mx_inv_sqrt = sp.diags(np.power(rowsums, -0.5))  # D**(-1/2)
    adj_mx_norm = (
        adj_mx_.dot(degree_mx_inv_sqrt).transpose().dot(degree_mx_inv_sqrt).tocoo()
    )  # D**(-1/2)*A*D**(-1/2)
    return adj_mx_norm


def sample_neg_test_edges():
    return 1


def sparse2tuple(sparse_mx):
    """
    Extract value coordinates (as tuple), values and shape from a sparse matrix.
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    value_coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return value_coords, values, shape


def train_test_split(
    adata, adj_mx_key: str = "spatial_connectivities", frac_train=0.85
):
    """
    Splits adata into training and test data
    """

    frac_test = 1 - frac_train

    adj_mx = adata.obsp[adj_mx_key]
    adj_mx_triu = sp.triu(adj_mx)  # upper triangle of adjacency matrix

    edges_single = sparse2tuple(adj_mx_triu)  # single edge for adjacent cells
    # edges_double = sparse2tuple(adj_mx)  # double edge for adjacent cells

    n_test = int(np.floor(edges_single.shape[0] * frac_test))
    idx_edges_all = np.array(range(edges_single.shape[0]))
    np.random.shuffle(idx_edges_all)
    idx_edges_test = idx_edges_all[:n_test]
    edges_test = edges_single[idx_edges_test]
    edges_train = np.delete(edges_single, idx_edges_test, axis=0)

    adj_train = sp.csr_matrix(
        (np.ones(edges_train.shape[0]), (edges_train[:, 0], edges_train[:, 1])),
        shape=adj_mx.shape,
    )
    adj_train = adj_train + adj_train.T
    adj_test = sp.csr_matrix(
        (np.ones(edges_test.shape[0]), (edges_test[:, 0], edges_test[:, 1])),
        shape=adj_mx.shape,
    )
    adj_test = adj_test + adj_test.T
    return adj_train, adj_test, edges_train, edges_test


def make_dataset(
    adata,
    frac_train,
):
    """
    Splits adata into train and validation data.

    Parameters
    ----------
    """
    return 1
