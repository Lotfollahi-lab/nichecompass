import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def sparse_mx_to_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_spatial_adata_from_csv(
        x_file_path,
        adj_file_path,
        adj_key="spatial_connectivities"):
    """
    Create AnnData object from csv files containing gene expression and 
    adjacency matrix.
    """
    adata = ad.read_csv(x_file_path)
    adj_df = pd.read_csv(adj_file_path, sep=",", header = 0)
    adj = adj_df.values
    adata.obsp[adj_key] = sp.csr_matrix(adj).tocoo()

    return adata