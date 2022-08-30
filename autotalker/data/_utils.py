import os
from typing import Optional

import numpy as np
import pyreadr
import torch
from scipy.sparse import csr_matrix


def _sparse_mx_to_sparse_tensor(sparse_mx: csr_matrix):
    """
    Helper to convert a scipy sparse matrix into a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

    