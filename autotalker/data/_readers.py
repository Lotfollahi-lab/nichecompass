import anndata as ad
import pandas as pd
import scipy.sparse as sp


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