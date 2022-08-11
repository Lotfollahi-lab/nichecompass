import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from ._utils import sparse_mx_to_sparse_tensor


class SpatialAnnDataset():
    def __init__(
            self,
            adata: ad.AnnData,
            adj_key: str = "spatial_connectivities"):

        # Store features in dense format
        if sp.issparse(adata.X): 
            self.x = torch.FloatTensor(adata.X.toarray())
        else:
            self.x = torch.FloatTensor(adata.X)

        self.n_node_features = self.x.size(1)

        # Store adjacency matrix in sparse tensor format
        self.adj = sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        if not (self.adj.to_dense() == self.adj.to_dense().T).all():
            raise ImportError("The input adjacency matrix is not symmetric.")

        self.edge_index = self.adj._indices()