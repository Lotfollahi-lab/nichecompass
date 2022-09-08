import anndata as ad
import pandas as pd
import scipy.sparse as sp


def load_spatial_adata_from_csv(x_file_path: str,
                                adj_file_path: str,
                                adj_key: str="spatial_connectivities"):
    """
    Create AnnData object from two csv files containing gene expression feature 
    matrix and adjacency matrix respectively.

    Parameters
    ----------
    x_file_path:
        File path of the csv file which contains gene expression feature matrix
        data.
    adj_file_path:
        File path of the csv file which contains adjacency matrix data.
    adj_key:
        Key under which the sparse adjacency matrix will be stored in 
        ´adata.obsp´.

    Returns
    ----------
    adata:
        AnnData object with gene expression data stored in ´adata.X´ and sparse 
        adjacency matrix (coo format) stored in ´adata.obps[adj_key]´.
    """
    adata = ad.read_csv(x_file_path)
    adj_df = pd.read_csv(adj_file_path, sep=",", header=0)
    adj = adj_df.values
    adata.obsp[adj_key] = sp.csr_matrix(adj).tocoo()
    return adata