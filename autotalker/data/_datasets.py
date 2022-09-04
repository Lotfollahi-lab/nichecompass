from typing import Literal, Optional

import scipy.sparse as sp
import torch
from anndata import AnnData

from ._utils import _sparse_mx_to_sparse_tensor


class SpatialAnnTorchDataset():
    """
    Spatially annotated torch dataset class to extract node features, node 
    labels, adjacency matrix and edge indices in a standardized format from an 
    AnnData object.

    Parameters
    ----------
    adata:
        AnnData object with sparse adjacency matrix stored in 
        adata.obsp[adj_key].
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    node_label_method:
        Node label method that will be used for gene expression reconstruction. 
        If ´self´, use only the input features of the node itself as node labels
        for gene expression reconstruction. If ´one-hop´, use a concatenation of
        the node's input features with a sum of the input features of all nodes
        in the node's one-hop neighborhood.
    """
    def __init__(self,
                 adata: AnnData,
                 adj_key: str="spatial_connectivities",
                 node_label_method: Literal["self", "one-hop"]="one-hop"):
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
        
        # Store labels for gene expression reconstruction (the node's own gene
        # expression concatenated with the node's neighbors feature expression
        # summed)
        x_neighbors_summed = torch.matmul(self.adj, self.x)
        self.x_one_hop = torch.cat((self.x, x_neighbors_summed), dim=-1)

        if node_label_method == "self":
            self.node_labels = self.x
        elif node_label_method == "one-hop":
            self.node_labels = self.x_one_hop

        self.edge_index = self.adj._indices()
        self.n_node_features = self.x.size(1)
        self.n_node_labels = self.node_labels.size(1)
        self.size_factors = self.x.sum(1)

    def __len__(self):
        return self.x.size(0)