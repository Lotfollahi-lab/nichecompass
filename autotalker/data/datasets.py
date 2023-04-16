"""
This module contains the SpatialAnnTorchDataset class to provide a standardized
dataset format for the training of an Autotalker model.
"""

from typing import Optional

import scipy.sparse as sp
import torch
from anndata import AnnData

from .utils import encode_labels, sparse_mx_to_sparse_tensor


class SpatialAnnTorchDataset():
    """
    Spatially annotated torch dataset class to extract node features, node 
    labels, adjacency matrix and edge indices in a standardized format from an 
    AnnData object.

    Parameters
    ----------
    adata:
        AnnData object with counts stored in ´adata.layers[counts_key]´ or
        ´adata.X´ depending on ´counts_key´, and sparse adjacency matrix stored
        in ´adata.obsp[adj_key]´.
    adata_atac:
        Additional optional AnnData object with paired spatial ATAC data.
    condition_label_encoder:
        Condition label encoder from the model (label encoding indeces need to
        be aligned with the ones from the model to get the correct conditional
        embedding).
    counts_key:
        Key under which the counts are stored in ´adata.layer´. If ´None´, uses
        ´adata.X´ as counts. 
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    condition_key:
        Key under which the condition for the conditional embedding is stored in
        ´adata.obs´.
    """
    def __init__(self,
                 adata: AnnData,
                 condition_label_encoder: dict,
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 condition_key: Optional[str]=None):
        if counts_key is None:
            x = adata.X
        else:
            x = adata.layers[counts_key]

        # Store features in dense format
        if sp.issparse(x): 
            self.x = torch.tensor(x.toarray())
        else:
            self.x = torch.tensor(x)

        # Store ATAC feature vector in dense format if provided
        if adata_atac is not None:
            if sp.issparse(adata_atac.X): 
                self.x_atac = torch.tensor(adata_atac.X.toarray())
            else:
                self.x_atac = torch.tensor(adata_atac.X)


        # Store adjacency matrix in torch_sparse SparseTensor format
        if sp.issparse(adata.obsp[adj_key]):
            self.adj = sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        else:
            self.adj = sparse_mx_to_sparse_tensor(
                sp.csr_matrix(adata.obsp[adj_key]))

        # Validate adjacency matrix symmetry
        if (self.adj.nnz() != self.adj.t().nnz()):
            raise ImportError("The input adjacency matrix has to be symmetric.")
        
        self.edge_index = self.adj.to_torch_sparse_coo_tensor()._indices()

        if condition_key is not None:
            self.conditions = torch.tensor(
                encode_labels(adata,
                              condition_label_encoder,
                              condition_key), dtype=torch.long)

        self.n_node_features = self.x.size(1)
        self.size_factors = self.x.sum(1) # fix for ATAC case

    def __len__(self):
        """Return the number of observations stored in SpatialAnnTorchDataset"""
        return self.x.size(0)