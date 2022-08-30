import torch_geometric


def node_level_split_mask(data: torch_geometric.data.Data,
                          val_ratio: float=0.1,
                          test_ratio: float=0.0):
    """
    Split data into training, validation and test sets on node-level by adding
    node-level masks (train_mask, val_mask, test_mask) to the PyG Data object.

    Parameters
    ----------
    data:
        PyG Data object to be split.
    valid_ratio:
        Ratio of edges to be included in validation split.
    test_ratio:
        Ratio of edges to be included in test split.
    Returns
    ----------
    data:
        PyG Data object with train_mask, val_mask and test_mask attributes 
        added.
    """
    node_split = torch_geometric.transforms.RandomNodeSplit(num_val=val_ratio,
                                                            num_test=test_ratio,
                                                            key="x")

    data = node_split(data)

    return data
    

def edge_level_split(data: torch_geometric.data.Data,
                     val_ratio: float=0.1,
                     test_ratio: float=0.1,
                     is_undirected: bool=True,
                     neg_sampling_ratio: float=0.0):
    """
    Split PyG Data object into train, validation and test PyG Data objects using
    a link/edge level split, i.e. training split does not include edges in 
    validation and test splits; and the validation split does not include edges
    in the test split. However, nodes will not be split and all node features 
    will be accessible from all splits.

    Parameters
    ----------
    data:
        PyG Data object to be split.
    valid_ratio:
        Ratio of edges to be included in validation split.
    test_ratio:
        Ratio of edges to be included in test split.
    is_undirected:
        If ´True´ only include 1 edge index pair per edge in edge_label_index
        and exclude symmetric edge index pair.
    neg_sampling_ratio:
        Ratio of negative sampling. This should be set to 0 if negative sampling
        is done by the dataloader.

    Returns
    ----------
    train_data:
        Training PyG Data object.
    valid_data:
        Validation PyG Data object.
    test_data:
        Test PyG Data object.
    """            
    
    link_split = torch_geometric.transforms.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=is_undirected, 
        neg_sampling_ratio=neg_sampling_ratio)

    train_data, val_data, test_data = link_split(data)

    return train_data, val_data, test_data
