from typing import Literal

from anndata import AnnData
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.transforms import RandomLinkSplit

from ._datasets import SpatialAnnTorchDataset


def edge_level_split(data: Data,
                     val_ratio: float=0.1,
                     test_ratio: float=0.05,
                     is_undirected: bool=True,
                     neg_sampling_ratio: float=0.0):
    """
    Split a PyG Data object into training, validation and test PyG Data objects 
    using an edge-level split, i.e. training split does not include edges in 
    validation and test splits; and the validation split does not include edges
    in the test split. However, nodes will not be split and all node features 
    will be accessible from all splits.

    Parameters
    ----------
    data:
        PyG Data object to be split.
    val_ratio:
        Ratio of edges to be included in the validation split.
    test_ratio:
        Ratio of edges to be included in the test split.
    is_undirected:
        If ´True´, only include 1 edge index pair per edge in edge_label_index
        and exclude symmetric edge index pair.
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
    
    random_link_split = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=is_undirected, 
        neg_sampling_ratio=neg_sampling_ratio)
    train_data, val_data, test_data = random_link_split(data)
    return train_data, val_data, test_data


def node_level_split_mask(data: Data,
                          val_ratio: float=0.1,
                          test_ratio: float=0.0,
                          split_key: str="x"):
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
                 adj_key: str="spatial_connectivities",
                 node_label_method: Literal["self", "one-hop"]="one-hop",
                 edge_val_ratio: float=0.1,
                 edge_test_ratio: float=0.05,
                 node_val_ratio: float=0.1,
                 node_test_ratio: float=0.0):
    """
    Prepare data for model training including edge-level and node-level train, 
    validation, test splits.

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
        the node's input features with an average of the input features of all 
        nodes in the node's one-hop neighborhood.
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
    train_data:
        PyG Data object containing training edges in the ´edge_label_index´ 
        attribute and training edge labels in the ´edge_label´ attribute.
    val_data:
        PyG Data object containing validation edges in the ´edge_label_index´ 
        attribute and validation edge labels in the ´edge_label´ attribute.
    test_data:
        PyG Data object containing test edges in the ´edge_label_index´ 
        attribute and test edge labels in the ´edge_label´ attribute.
    """
    data_dict = {}
    dataset = SpatialAnnTorchDataset(adata=adata,
                                     adj_key=adj_key,
                                     node_label_method=node_label_method)
    # PyG Data object (has 2 edge index pairs for one edge because of symmetry)
    data = Data(x=dataset.x,
                edge_index=dataset.edge_index,
                node_labels=dataset.node_labels)

    # Edge-level split for edge reconstruction
    edge_train_data, edge_val_data, edge_test_data = edge_level_split(
        data=data,
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