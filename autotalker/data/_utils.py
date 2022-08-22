import os

import numpy as np
import pyreadr
import scipy.sparse as sp
import torch
from typing import Optional


def sparse_mx_to_sparse_tensor(sparse_mx: sp.csr_matrix):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def load_R_file_as_df(R_file_path: str,
                      url: Optional[str]=None,
                      save_df_to_disk: bool=False,
                      df_save_path: Optional[str]=None):
    """
    Load R file either from ´url´ if specified or from ´file_path´ on disk
    and convert to pandas dataframe.
    """
    if url is None:
        if not os.path.exists(R_file_path):
            raise ValueError("Please specify a valid ´file_path´ or ´url´.")
        result_odict = pyreadr.read_r(R_file_path)
    else:
        result_odict = pyreadr.read_r(pyreadr.download_file(url, R_file_path))
        os.remove(R_file_path)

    df = result_odict[None]

    if save_df_to_disk:
        if df_save_path == None:
            raise ValueError("Please specify ´df_save_path´ or set " 
                             "´save_to_disk.´ to False")
        df.to_csv(df_save_path)

    return df

    