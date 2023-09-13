"""
This module contains the SpatialAnnTorchDataset class to provide a standardized
dataset format for the training of an NicheCompass model.
"""

from typing import List, Optional

import scipy.sparse as sp
import torch
from anndata import AnnData
from torch_geometric.utils import add_self_loops, remove_self_loops

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
    cat_covariates_label_encoders:
        List of categorical covariates label encoders from the model (label
        encoding indeces need to be aligned with the ones from the model to get
        the correct categorical covariates embeddings).
    counts_key:
        Key under which the counts are stored in ´adata.layer´. If ´None´, uses
        ´adata.X´ as counts. 
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    self_loops:
        If ´True´, add self loops to the adjacency matrix to model autocrine
        communication.
    cat_covariates_keys:
        Keys under which the categorical covariates are stored in ´adata.obs´.
    """
    def __init__(self,
                 adata: AnnData,
                 cat_covariates_label_encoders: List[dict],
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 edge_label_adj_key: str="edge_label_spatial_connectivities",
                 self_loops: bool=True,
                 cat_covariates_keys: Optional[str]=None):
        if counts_key is None:
            x = adata.X
        else:
            x = adata.layers[counts_key]

        # Store features in dense format
        if sp.issparse(x): 
            self.x = torch.tensor(x.toarray())
        else:
            self.x = torch.tensor(x)

        # Concatenate ATAC feature vector in dense format if provided
        if adata_atac is not None:
            if sp.issparse(adata_atac.X): 
                self.x = torch.cat(
                    (self.x, torch.tensor(adata_atac.X.toarray())), axis=1)
            else:
                self.x = torch.cat((self.x, torch.tensor(adata_atac.X)), axis=1)            

        # Store adjacency matrix in torch_sparse SparseTensor format
        if sp.issparse(adata.obsp[adj_key]):
            self.adj = sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        else:
            self.adj = sparse_mx_to_sparse_tensor(
                sp.csr_matrix(adata.obsp[adj_key]))
            
        # Store edge label adjacency matrix
        if edge_label_adj_key in adata.obsp:
            self.edge_label_adj = sp.csr_matrix(adata.obsp[edge_label_adj_key])
        else:
            self.edge_label_adj = None

        # Validate adjacency matrix symmetry
        if (self.adj.nnz() != self.adj.t().nnz()):
            raise ImportError("The input adjacency matrix has to be symmetric.")
        
        self.edge_index = self.adj.to_torch_sparse_coo_tensor()._indices()

        if self_loops:
            # Add self loops to account for autocrine communication
            # Remove self loops in case there are already before adding new ones
            self.edge_index, _ = remove_self_loops(self.edge_index)
            self.edge_index, _ = add_self_loops(self.edge_index,
                                                num_nodes=self.x.size(0))
            
        if cat_covariates_keys is not None:
            self.cat_covariates_cats = []
            for cat_covariate_key, cat_covariate_label_encoder in zip(
                cat_covariates_keys,
                cat_covariates_label_encoders):
                cat_covariate_cats = torch.tensor(
                    encode_labels(adata,
                                  cat_covariate_label_encoder,
                                  cat_covariate_key), dtype=torch.long)
                self.cat_covariates_cats.append(cat_covariate_cats)
            self.cat_covariates_cats = torch.stack(self.cat_covariates_cats,
                                                   dim=1)            

        self.n_node_features = self.x.size(1)
        self.size_factors = self.x.sum(1) # fix for ATAC case

    def __len__(self):
        """Return the number of observations stored in SpatialAnnTorchDataset"""
        return self.x.size(0)