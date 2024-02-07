"""
This module contains data readers for the training of an NicheCompass model.
"""

from typing import Optional

import anndata as ad
import pandas as pd
import scipy.sparse as sp


def load_spatial_adata_from_csv(counts_file_path: str,
                                adj_file_path: str,
                                cell_type_file_path: Optional[str]=None,
                                adj_key: str="spatial_connectivities",
                                cell_type_col: str="cell_type",
                                cell_type_key: str="cell_type") -> ad.AnnData:
    """
    Create AnnData object from two csv files containing gene expression feature 
    matrix and adjacency matrix respectively. Optionally, a third csv file with
    cell types can be provided.

    Parameters
    ----------
    counts_file_path:
        File path of the csv file which contains gene expression feature matrix
        data.
    adj_file_path:
        File path of the csv file which contains adjacency matrix data.
    cell_type_file_path:
        File path of the csv file which contains cell type data.
    adj_key:
        Key under which the sparse adjacency matrix will be stored in 
        ´adata.obsp´.
    cell_type_col:
        Column under wich the cell type is stored in the ´cell_type_file´.
    cell_type_key:
        Key under which the cell types will be stored in ´adata.obs´.

    Returns
    ----------
    adata:
        AnnData object with gene expression data stored in ´adata.X´ and sparse 
        adjacency matrix (coo format) stored in ´adata.obps[adj_key]´.
    """
    adata = ad.read_csv(counts_file_path)
    adj_df = pd.read_csv(adj_file_path, sep=",", header=0)
    adj = adj_df.values
    adata.obsp[adj_key] = sp.csr_matrix(adj).tocoo()

    if cell_type_file_path:
        cell_type_df = pd.read_csv(cell_type_file_path, sep=",", header=0)
        adata.obs[cell_type_key] = cell_type_df[cell_type_col].values
    return adata