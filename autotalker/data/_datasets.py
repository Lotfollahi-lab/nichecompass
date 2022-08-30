from typing import Optional

import scipy.sparse as sp
import torch
from anndata import AnnData

from ._utils import _sparse_mx_to_sparse_tensor


class SpatialAnnTorchDataset():
    """
    Spatially annotated torch dataset class to extract node features, adjacency 
    matrix and edge indices in a standardized format from an AnnData object.

    Parameters
    ----------
    adata:
        AnnData object with sparse adjacency matrix stored in 
        adata.obsp[adj_key].
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    """
    def __init__(self,
                 adata: AnnData,
                 adj_key: str="spatial_connectivities"):
        # Store features in dense format
        if sp.issparse(adata.X): 
            self.x = torch.tensor(adata.X.toarray())
        else:
            self.x = torch.tensor(adata.X)

        # Store adjacency matrix in sparse torch tensor format
        if sp.issparse(adata.obsp[adj_key]):
            self.adj = _sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        else:
            self.adj = _sparse_mx_to_sparse_tensor(
                sp.csr_matrix(adata.obsp[adj_key]))

        # Validate adjacency matrix symmetry
        if not (self.adj.to_dense() == self.adj.to_dense().T).all():
            raise ImportError("The input adjacency matrix has to be symmetric.")

        self.edge_index = self.adj._indices()
        self.n_node_features = self.x.size(1)
        self.size_factors = self.x.sum(1)

    def __len__(self):
        return self.x.size(0)