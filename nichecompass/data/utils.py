"""
This module contains helper functions for the ´data´ subpackage.
"""

import anndata as ad
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor


def encode_labels(adata: ad.AnnData,
                  label_encoder: dict,
                  label_key="str") -> np.ndarray:
    """
    Encode labels from an `adata` object stored in `adata.obs` to integers.

    Implementation is adapted from
    https://github.com/theislab/scarches/blob/c21492d409150cec73d26409f9277b3ac971f4a7/scarches/dataset/trvae/_utils.py#L4
    (20.01.2023).

    Parameters
    ----------
    adata:
        AnnData object with labels stored in `adata.obs[label_key]`.
    label_encoder:
        Dictionary where keys are labels and values are label encodings.
    label_key:
        Key where in `adata.obs` the labels to be encoded are stored.

    Returns
    -------
    encoded_labels:
        Integer-encoded labels.
    """
    unique_labels = list(np.unique(adata.obs[label_key]))
    encoded_labels = np.zeros(adata.shape[0])

    if not set(unique_labels).issubset(set(label_encoder.keys())):
        print(f"Warning: Labels in adata.obs[{label_key}] are not a subset of "
              "the label encoder!")
        print("Therefore integer value of those labels is set to '-1'.")
        for unique_label in unique_labels:
            if unique_label not in label_encoder.keys():
                encoded_labels[adata.obs[label_key] == unique_label] = -1

    for label, label_encoding in label_encoder.items():
        encoded_labels[adata.obs[label_key] == label] = label_encoding
    return encoded_labels

    
def sparse_mx_to_sparse_tensor(sparse_mx: csr_matrix) -> SparseTensor:
    """
    Convert a scipy sparse matrix into a torch_sparse SparseTensor.

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