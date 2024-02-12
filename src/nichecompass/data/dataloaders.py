"""
This module contains dataloaders for the training of an NicheCompass model.
"""

from typing import Optional

from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader


def initialize_dataloaders(node_masked_data: Data,
                           edge_train_data: Optional[Data]=None,
                           edge_val_data: Optional[Data]=None,
                           edge_batch_size: Optional[int]=64,
                           node_batch_size: int=64,
                           n_direct_neighbors: int=-1,
                           n_hops: int=1,
                           shuffle: bool=True,
                           edges_directed: bool=False,
                           neg_edge_sampling_ratio: float=1.) -> dict:
    """
    Initialize edge-level and node-level training and validation dataloaders.

    Parameters
    ----------
    node_masked_data:
        PyG Data object with node-level split masks.
    edge_train_data:
        PyG Data object containing the edge-level training set.
    edge_val_data:
        PyG Data object containing the edge-level validation set.
    edge_batch_size:
        Batch size for the edge-level dataloaders.
    node_batch_size:
        Batch size for the node-level dataloaders.
    n_direct_neighbors:
        Number of sampled direct neighbors of the current batch nodes to be 
        included in the batch. Defaults to ´-1´, which means to include all 
        direct neighbors.
    n_hops:
        Number of neighbor hops / levels for neighbor sampling of nodes to be 
        included in the current batch. E.g. ´2´ means to not only include 
        sampled direct neighbors of current batch nodes but also sampled 
        neighbors of the direct neighbors.
    shuffle:
        If `True`, shuffle the dataloaders.
    edges_directed:
        If `False`, both symmetric edge index pairs are included in the same 
        edge-level batch (1 edge has 2 symmetric edge index pairs).
    neg_edge_sampling_ratio:
        Negative sampling ratio of edges. This is currently implemented in an
        approximate way, i.e. negative edges may contain false negatives.

    Returns
    ----------
    loader_dict:
        Dictionary containing training and validation PyG LinkNeighborLoader 
        (for edge reconstruction) and NeighborLoader (for gene expression 
        reconstruction) objects.
    """
    loader_dict = {}

    # Node-level dataloaders
    loader_dict["node_train_loader"] = NeighborLoader(
        node_masked_data,
        num_neighbors=[n_direct_neighbors] * n_hops,
        batch_size=node_batch_size,
        directed=False,
        shuffle=shuffle,
        input_nodes=node_masked_data.train_mask)
    if node_masked_data.val_mask.sum() != 0:
        loader_dict["node_val_loader"] = NeighborLoader(
            node_masked_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=node_batch_size,
            directed=False,
            shuffle=shuffle,
            input_nodes=node_masked_data.val_mask)
        
    # Edge-level dataloaders
    if edge_train_data is not None:
        loader_dict["edge_train_loader"] = LinkNeighborLoader(
            edge_train_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label=None, # will automatically be added as 1 for all edges
            edge_label_index=edge_train_data.edge_label_index[:, edge_train_data.edge_label.bool()], # limit the edges to the ones from the edge_label_adj
            directed=edges_directed,
            shuffle=shuffle,
            neg_sampling_ratio=neg_edge_sampling_ratio)
    if edge_val_data is not None and edge_val_data.edge_label.sum() != 0:
        loader_dict["edge_val_loader"] = LinkNeighborLoader(
            edge_val_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label=None, # will automatically be added as 1 for all edges
            edge_label_index=edge_val_data.edge_label_index[:, edge_val_data.edge_label.bool()], # limit the edges to the ones from the edge_label_adj
            directed=edges_directed,
            shuffle=shuffle,
            neg_sampling_ratio=neg_edge_sampling_ratio)

    return loader_dict