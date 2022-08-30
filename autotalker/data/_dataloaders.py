from typing import Optional

from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.loader import NeighborLoader


def initialize_dataloaders(node_masked_data: Data,
                           edge_train_data: Data,
                           edge_val_data: Optional[Data]=None,
                           edge_test_data: Optional[Data]=None,
                           node_batch_size: int=4,
                           edge_batch_size: int=32,
                           n_direct_neighbors: int=-1,
                           n_hops: int=3,
                           edges_directed: bool=False,
                           neg_edge_sampling_ratio: float=1.0):
    """
    Initialize edge-level and node-level training, validation and test 
    dataloaders.

    Parameters
    ----------
    node_masked_data:
        PyG Data object with node-level split masks.
    edge_train_data:
        PyG Data object containing the edge-level training set.
    edge_val_data:
        PyG Data object containing the edge-level validation set.
    node_batch_size:
        Batch size for the node-level dataloaders.
    edge_batch_size:
        Batch size for the edge-level dataloaders.
    n_direct_neighbors:
        Number of direct neighbors of the sampled nodes to be included in the 
        batch. Defaults to ´-1´, which means to include all direct neighbors.
    n_hops:
        Number of neighbor hops/levels to be included in the batch. E.g. ´2´
        means to not only include direct neighbors of sampled nodes but also the
        neighbors of the direct neighbors.
    edges_directed:
        If `False` both symmetric edge index pairs are included in the same 
        edge-level batch (1 edge has 2 symmetric edge index pairs).
    neg_edge_sampling_ratio:
        Negative sampling ratio of edges. This is currently implemented in an
        approximate way, i.e. negative edges may contain false negatives.

    Returns
    ----------
    loader_dict:
        Dictionary containing training, validation and test PyG 
        LinkNeighborLoader (for edge reconstruction) and NeighborLoader (for
        gene expression reconstruction) objects.
    """
    loader_dict = {}

    loader_dict["node_train_loader"] = NeighborLoader(
        node_masked_data,
        num_neighbors=[n_direct_neighbors] * n_hops,
        batch_size=node_batch_size,
        directed=False,
        shuffle=True,
        input_nodes=node_masked_data.train_mask)
        
    if node_masked_data.val_mask.sum() != 0:
        loader_dict["node_val_loader"] = NeighborLoader(
            node_masked_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=node_batch_size,
            directed=False,
            shuffle=True,
            input_nodes=node_masked_data.val_mask)
            
    if node_masked_data.test_mask.sum() != 0:
        loader_dict["node_test_loader"] = NeighborLoader(
            node_masked_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=node_batch_size,
            directed=False,
            shuffle=True,
            input_nodes=node_masked_data.test_mask)

    loader_dict["edge_train_loader"] = LinkNeighborLoader(
        edge_train_data,
        num_neighbors=[n_direct_neighbors] * n_hops,
        batch_size=edge_batch_size,
        edge_label_index=edge_train_data.edge_label_index,
        directed=edges_directed, 
        neg_sampling_ratio=neg_edge_sampling_ratio)

    if edge_val_data.edge_label.sum() != 0:
        loader_dict["edge_val_loader"] = LinkNeighborLoader(
            edge_val_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label_index=edge_val_data.edge_label_index,
            directed=edges_directed, 
            neg_sampling_ratio=neg_edge_sampling_ratio)
            
    if edge_test_data.edge_label.sum() != 0:
        loader_dict["edge_test_loader"] = LinkNeighborLoader(
            edge_test_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label_index=edge_test_data.edge_label_index,
            directed=edges_directed, 
            neg_sampling_ratio=neg_edge_sampling_ratio)

    return loader_dict