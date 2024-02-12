"""
This module contains data processors for the training of an NicheCompass model.
"""

from typing import List, Optional, Tuple

import scipy.sparse as sp
import torch
from anndata import AnnData
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit
from torch_geometric.utils import add_self_loops, remove_self_loops

from .datasets import SpatialAnnTorchDataset


def edge_level_split(data: Data,
                     edge_label_adj: Optional[sp.csr_matrix],
                     val_ratio: float=0.1,
                     test_ratio: float=0.,
                     is_undirected: bool=True,
                     neg_sampling_ratio: float=0.) -> Tuple[Data, Data, Data]:
    """
    Split a PyG Data object into training, validation and test PyG Data objects 
    using an edge-level split. The training split does not include edges in the
    validation and test splits and the validation split does not include edges
    in the test split. However, nodes will not be split and all node features 
    will be accessible from all splits.

    Check https://github.com/pyg-team/pytorch_geometric/issues/3668 for more
    context how RandomLinkSplit works.

    Parameters
    ----------
    data:
        PyG Data object to be split.
    edge_label_adj:
        Adjacency matrix which contains edges for edge reconstruction. If 
        ´None´, uses the 'normal' adjacency matrix used for message passing.
    val_ratio:
        Ratio of edges to be included in the validation split.
    test_ratio:
        Ratio of edges to be included in the test split.
    is_undirected:
        If ´True´, the graph is assumed to be undirected, and positive and
        negative samples will not leak (reverse) edge connectivity across
        different splits. This is set to ´False´, as there is an issue with
        replication of self loops.
    neg_sampling_ratio:
        Ratio of negative sampling. This should be set to 0 if negative sampling
        is done by the dataloader.

    Returns
    ----------
    train_data:
        Training PyG Data object.
    val_data:
        Validation PyG Data object.
    test_data:
        Test PyG Data object.
    """
    # Clone data to not modify in place to not affect node split
    data_no_self_loops = data.clone()

    # Remove self loops temporarily as we don't want them as edge labels. There
    # is also an issue with RandomLinkSplit (self loops will be replicated for
    # message passing). We will add the self loops again after the split
    data_no_self_loops.edge_index, data_no_self_loops.edge_attr = (
        remove_self_loops(edge_index=data.edge_index,
                          edge_attr=data.edge_attr))
    
    if edge_label_adj is not None:
        # Add edge label which is 1 for edges from edge_label_adj and 0 otherwise.
        # This will be used by dataloader to only sample edges from edge_label_adj
        # as opposed to from adj.
        data_no_self_loops.edge_label = torch.tensor(
            [(edge_label_adj[edge_index[0].item(), edge_index[1].item()] == 1.0) for
             edge_index in data_no_self_loops.edge_attr]).int()
    
    random_link_split = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=is_undirected,
        key="edge_label", # if ´edge_label´ is not existent, it will be added with 1s
        neg_sampling_ratio=neg_sampling_ratio)
    train_data, val_data, test_data = random_link_split(data_no_self_loops)
    
    # Readd self loops for message passing
    for split_data in [train_data, val_data, test_data]:
        split_data.edge_index = add_self_loops(
            edge_index=split_data.edge_index,
            num_nodes=split_data.x.shape[0])[0]
        split_data.edge_attr = add_self_loops(
            edge_index=split_data.edge_attr.t(),
            num_nodes=split_data.x.shape[0])[0].t()
    return train_data, val_data, test_data


def node_level_split_mask(data: Data,
                          val_ratio: float=0.1,
                          test_ratio: float=0.,
                          split_key: str="x") -> Data:
    """
    Split data on node-level into training, validation and test sets by adding
    node-level masks (train_mask, val_mask, test_mask) to the PyG Data object.

    Parameters
    ----------
    data:
        PyG Data object to be split.
    val_ratio:
        Ratio of nodes to be included in the validation split.
    test_ratio:
        Ratio of nodes to be included in the test split.
    split_key:
        The attribute key of the PyG Data object that holds the ground
        truth labels. Only nodes in which the key is present will be split.

    Returns
    ----------
    data:
        PyG Data object with ´train_mask´, ´val_mask´ and ´test_mask´ attributes 
        added.
    """
    random_node_split = RandomNodeSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        key=split_key)
    data = random_node_split(data)
    return data


def prepare_data(adata: AnnData,
                 cat_covariates_label_encoders: List[dict],
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 cat_covariates_keys: Optional[List[str]]=None,
                 edge_val_ratio: float=0.1,
                 edge_test_ratio: float=0.,
                 node_val_ratio: float=0.1,
                 node_test_ratio: float=0.) -> dict:
    """
    Prepare data for model training including edge-level and node-level train, 
    validation, and test splits.

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
    cat_covariates_keys:
        Keys under which the categorical covariates are stored in ´adata.obs´.
    edge_val_ratio:
        Fraction of the data that is used as validation set on edge-level.
    edge_test_ratio:
        Fraction of the data that is used as test set on edge-level.
    node_val_ratio:
        Fraction of the data that is used as validation set on node-level.
    node_test_ratio:
        Fraction of the data that is used as test set on node-level.

    Returns
    ----------
    data_dict:
        Dictionary containing edge-level training, validation and test PyG 
        Data objects and node-level PyG Data object with split masks under keys 
        ´edge_train_data´, ´edge_val_data´, ´edge_test_data´, and 
        ´node_masked_data´ respectively. The edge-level PyG Data objects contain
        edges in the ´edge_label_index´ attribute and edge labels in the 
        ´edge_label´ attribute.
    """
    data_dict = {}
    dataset = SpatialAnnTorchDataset(
        adata=adata,
        adata_atac=adata_atac,
        counts_key=counts_key,
        adj_key=adj_key,
        cat_covariates_keys=cat_covariates_keys,
        cat_covariates_label_encoders=cat_covariates_label_encoders)

    # PyG Data object (has 2 edge index pairs for one edge because of symmetry;
    # one edge index pair will be removed in the edge-level split).
    data = Data(x=dataset.x,
                edge_index=dataset.edge_index,
                edge_attr=dataset.edge_index.t()) # store index of edge nodes as
                                                  # edge attribute for
                                                  # aggregation weight retrieval
                                                  # in mini batches

    if cat_covariates_keys is not None:
        data.cat_covariates_cats = dataset.cat_covariates_cats

    # Edge-level split for edge reconstruction
    edge_train_data, edge_val_data, edge_test_data = edge_level_split(
        data=data,
        edge_label_adj=dataset.edge_label_adj,
        val_ratio=edge_val_ratio,
        test_ratio=edge_test_ratio)
    data_dict["edge_train_data"] = edge_train_data
    data_dict["edge_val_data"] = edge_val_data
    data_dict["edge_test_data"] = edge_test_data

    # Node-level split for gene expression reconstruction
    data_dict["node_masked_data"] = node_level_split_mask(
        data=data,
        val_ratio=node_val_ratio,
        test_ratio=node_test_ratio)
    return data_dict