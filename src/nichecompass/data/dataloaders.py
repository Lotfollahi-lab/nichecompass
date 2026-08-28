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
                           neg_edge_sampling_ratio: float=1.,
                           node_input_shard_fn=None,
                           edge_input_shard_fn=None,
                           drop_last: bool=False) -> dict:
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
    node_input_shard_fn:
        Optional callable that receives the 1D tensor of input node indices and
        returns the subset that the current process should use. Used for
        distributed training, where every process holds the full graph but
        seeds its neighbor sampling from a disjoint subset of the nodes. If
        ´None´, every node of the respective split is used, which is the single
        device behavior.
    edge_input_shard_fn:
        Optional callable that receives the ´[2, n_edges]´ edge label index and
        returns the subset of its columns that the current process should use.
        The same considerations as for ´node_input_shard_fn´ apply. Note that
        only the seed edges are split; the graph itself is never partitioned,
        since neighbor aggregation needs every node's true neighbors.
    drop_last:
        If ´True´, drop the last incomplete batch. Distributed training sets
        this so that every process performs the same number of optimizer
        steps, since a process that ran out of batches early would leave the
        others waiting forever on the next gradient reduction.

    Returns
    ----------
    loader_dict:
        Dictionary containing training and validation PyG LinkNeighborLoader 
        (for edge reconstruction) and NeighborLoader (for gene expression 
        reconstruction) objects.
    """
    loader_dict = {}

    def shard_nodes(mask):
        """Return the input nodes this process should seed sampling from."""
        if node_input_shard_fn is None:
            return mask
        # ´NeighborLoader´ accepts either a boolean mask or an index tensor,
        # and sharding is only meaningful on indices
        return node_input_shard_fn(mask.nonzero(as_tuple=False).view(-1))

    def shard_edges(edge_label_index):
        """Return the seed edges this process should sample around."""
        if edge_input_shard_fn is None:
            return edge_label_index
        return edge_input_shard_fn(edge_label_index)

    # Node-level dataloaders
    loader_dict["node_train_loader"] = NeighborLoader(
        node_masked_data,
        num_neighbors=[n_direct_neighbors] * n_hops,
        batch_size=node_batch_size,
        directed=False,
        shuffle=shuffle,
        drop_last=drop_last,
        input_nodes=shard_nodes(node_masked_data.train_mask))
    if node_masked_data.val_mask.sum() != 0:
        loader_dict["node_val_loader"] = NeighborLoader(
            node_masked_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=node_batch_size,
            directed=False,
            shuffle=shuffle,
            drop_last=drop_last,
            input_nodes=shard_nodes(node_masked_data.val_mask))
        
    # Edge-level dataloaders
    if edge_train_data is not None:
        loader_dict["edge_train_loader"] = LinkNeighborLoader(
            edge_train_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label=None, # will automatically be added as 1 for all edges
            edge_label_index=shard_edges(edge_train_data.edge_label_index[:, edge_train_data.edge_label.bool()]), # limit the edges to the ones from the edge_label_adj
            directed=edges_directed,
            shuffle=shuffle,
            drop_last=drop_last,
            neg_sampling_ratio=neg_edge_sampling_ratio)
    if edge_val_data is not None and edge_val_data.edge_label.sum() != 0:
        loader_dict["edge_val_loader"] = LinkNeighborLoader(
            edge_val_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label=None, # will automatically be added as 1 for all edges
            edge_label_index=shard_edges(edge_val_data.edge_label_index[:, edge_val_data.edge_label.bool()]), # limit the edges to the ones from the edge_label_adj
            directed=edges_directed,
            shuffle=shuffle,
            drop_last=drop_last,
            neg_sampling_ratio=neg_edge_sampling_ratio)

    return loader_dict