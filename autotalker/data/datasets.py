from typing import Literal

import scipy.sparse as sp
import torch
import torch_geometric
from anndata import AnnData

from .utils import _sparse_mx_to_sparse_tensor


class SpatialAnnTorchDataset():
    """
    Spatially annotated torch dataset class to extract node features, node 
    labels, adjacency matrix and edge indices in a standardized format from an 
    AnnData object.

    Parameters
    ----------
    adata:
        AnnData object with raw counts stored in 
        ´adata.layers[counts_layer_key]´, and sparse adjacency matrix stored in 
        ´adata.obsp[adj_key]´.
    counts_layer_key:
        Key under which the raw counts are stored in ´adata.layer´.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    node_label_method:
        Node label method that will be used for gene expression reconstruction. 
        If ´self´, use only the input features of the node itself as node labels
        for gene expression reconstruction. If ´one-hop-sum´, use a 
        concatenation of the node's input features with the sum of the input 
        features of all nodes in the node's one-hop neighborhood. If 
        ´one-hop-norm´, use a concatenation of the node`s input features with
        the node's one-hop neighbors input features normalized as per Kipf, T. 
        N. & Welling, M. Semi-Supervised Classification with Graph Convolutional
        Networks. arXiv [cs.LG] (2016))
    """
    def __init__(self,
                 adata: AnnData,
                 counts_layer_key: str="counts",
                 adj_key: str="spatial_connectivities",
                 node_label_method: Literal["self",
                                            "one-hop-sum",
                                            "one-hop-norm"]="one-hop-norm"):
        # Store features in dense format
        if sp.issparse(adata.layers[counts_layer_key]): 
            self.x = torch.tensor(adata.layers[counts_layer_key].toarray())
        else:
            self.x = torch.tensor(adata.layers[counts_layer_key])

        # Store adjacency matrix in torch_sparse SparseTensor format
        if sp.issparse(adata.obsp[adj_key]):
            self.adj = _sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        else:
            self.adj = _sparse_mx_to_sparse_tensor(
                sp.csr_matrix(adata.obsp[adj_key]))

        # Validate adjacency matrix symmetry
        if not (self.adj.to_dense() == self.adj.to_dense().T).all():
            raise ImportError("The input adjacency matrix has to be symmetric.")

        """
        # Store labels for gene expression reconstruction
        if node_label_method == "self":
            self.node_labels = self.x
        elif node_label_method == "one-hop-norm":
            gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
            adj_norm = gcn_norm(self.adj, add_self_loops=False)
            x_neighbors_norm = adj_norm.matmul(self.x)
            self.node_labels = torch.cat((self.x, x_neighbors_norm), dim=-1)
        elif node_label_method == "one-hop-sum":
            x_neighbors_sum = self.adj.matmul(self.x)
            self.node_labels = torch.cat((self.x, x_neighbors_sum), dim=-1)
        """

        self.edge_index = self.adj.to_torch_sparse_coo_tensor()._indices()
        self.n_node_features = self.x.size(1)
        # self.n_node_labels = self.node_labels.size(1)
        self.size_factors = self.x.sum(1)

    def __len__(self):
        return self.x.size(0)