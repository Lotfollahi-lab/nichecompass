import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor


def _sparse_mx_to_sparse_tensor(sparse_mx: csr_matrix):
    """
    Helper to convert a scipy sparse matrix into a torch_sparse SparseTensor.

    Parameters
    ----------
    sparse_mx:
        Sparse scipy csr_matrix.

    Returns
    ----------
    sparse_tensor:
        torch_sparse SparseTensor object that can be utilized by PyG.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    torch_sparse_coo_tensor = torch.sparse.FloatTensor(indices, values, shape)
    sparse_tensor = SparseTensor.from_torch_sparse_coo_tensor(
        torch_sparse_coo_tensor)
    return sparse_tensor
    