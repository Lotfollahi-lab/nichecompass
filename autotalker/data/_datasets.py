from typing import Optional

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from ._utils import sparse_mx_to_sparse_tensor
from ._utils import label_encoder


class SpatialAnnDataset():
    """
    SpatialAnnDataset class to extract node features, adjacency matrix and edge
    indices in a standardized format from an AnnData object.

    Parameters
    ----------
    adata:
        AnnData object with sparse adjacency matrix stored in 
        adata.obsp[adj_key].
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    """
    def __init__(self,
                 adata: ad.AnnData,
                 adj_key: str="spatial_connectivities",
                 size_factor_key: str="autotalker_size_factors",
                 condition_key: Optional[str]=None,
                 condition_label_dict: dict=None):
        self.condition_key = condition_key
        self.condition_label_dict = condition_label_dict
        
        # Store features in dense format
        if sp.issparse(adata.X): 
            self.x = torch.FloatTensor(adata.X.toarray())
        else:
            self.x = torch.FloatTensor(adata.X)

        # Store adjacency matrix in sparse tensor format
        if sp.issparse(adata.obsp[adj_key]):
            self.adj = sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        else:
            self.adj = sparse_mx_to_sparse_tensor(
                sp.csr_matrix(adata.obsp[adj_key]))

        # Validate adjacency matrix symmetry
        if not (self.adj.to_dense() == self.adj.to_dense().T).all():
            raise ImportError("The input adjacency matrix has to be symmetric.")

        # Add size factors
        size_factors = adata.X.sum(1)
        if len(size_factors.shape) < 2:
            size_factors = np.expand_dims(size_factors, axis=1)
        adata.obs["autotalker_size_factors"] = size_factors

        self.edge_index = self.adj._indices()
        self.n_node_features = self.x.size(1)
        self.size_factors = torch.FloatTensor(
            adata.obs["autotalker_size_factors"])

        # Encode condition strings to integer
        if self.condition_key is not None:
            self.conditions = label_encoder(
                adata,
                condition_label_dict=self.condition_label_dict,
                condition_key=condition_key)
            self.conditions = torch.tensor(self.conditions, dtype=torch.long)
        else:
            self.conditions = torch.zeros(len(adata.obs))

    @property
    def condition_label_dict(self) -> dict:
        return self.condition_label_dict

    @condition_label_dict.setter
    def condition_label_dict(self, value: dict):
        if value is not None:
            self.condition_label_dict = value