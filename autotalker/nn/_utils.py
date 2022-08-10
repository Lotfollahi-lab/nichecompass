import torch
import numpy as np
import scipy.sparse as sp


def sparse_matrix_to_sparse_tensor(sparse_mx):

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    values = sparse_mx.data
    indices = np.vstack((sparse_mx.row, sparse_mx.col))

    i = torch.IntTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_mx.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
