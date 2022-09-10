import torch
from torch_geometric.utils import to_dense_adj


def _edge_values_and_sorted_labels(adj: torch.Tensor,
                                   edge_label_index: torch.Tensor,
                                   edge_labels: torch.Tensor):
    """
    Utility function to retrieve values at edge indeces as given by 
    ´edge_label_index´ from an adjacency matrix ´adj´, remove edge labels 
    from ´edge_labels´ that are due to duplicate edge indeces (which can 
    happen because of the approximate negative sampling implementation in PyG 
    LinkNeighborLoader) and align the order of ´edge_labels´ with the values 
    retrieved from ´adj´.

    Parameters
    ----------
    adj:
        Tensor containing the values in adjacency matrix format. Values could
        be predicted logits or probabilities for example.
    edge_label_index:
        Tensor containing the indices of edges for which values should be
        retrieved from `adj`. Edge indices could be duplicate, in which case 
        the duplicate will be discarded and only one value will be retrieved
        from `adj`.
    edge_labels:
        Tensor containing the edge labels (0 for no edge / negative edge and 1
        for edge / positive edge). Can contain duplicate labels per edge due to 
        duplicate edge indices, in which case duplicates are removed from 
        `edge_labels`.
    
    Returns
    ----------
    adj_values:
        Tensor containing the values retrieved from `adj`.
    edge_labels_sorted:
        Tensor containing unique labels per edge sorted to match corresponding
        `adj_values`.
    """
    # Create mask to retrieve values at indeces as given in ´edge_label_index´ 
    # from ´adj´
    n_nodes = adj.shape[0]
    adj_labels = to_dense_adj(edge_label_index, max_num_nodes=n_nodes)
    mask = torch.squeeze(adj_labels > 0)
    
    # Retrieve values from ´adj´ (could be logits or probabilites depending on
    # what is stored in ´adj´)
    adj_values = torch.masked_select(adj, mask)
    
    # Sort ´edge_labels´ to align order with masked retrieval from ´adj´. In 
    # addition, remove entries in ´edge_labels´ that are due to duplicates in 
    # ´edge_label_index´, 
    sort_index = _unique_sorted_index(edge_label_index)
    edge_labels_sorted = edge_labels[sort_index]

    return adj_values, edge_labels_sorted


def _unique_sorted_index(x: torch.Tensor, dim=-1):
    """
    Utility function to remove duplicates from a tensor and return a sorted 
    index containing only indeces from unique values.

    Parameters
    ----------
    x:
        Tensor for which to return the unique sorted index.

    Returns
    ----------
    unique_sorted_index:
        Unique sorted index of the input tensor.
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    unique_sorted_index = inverse.new_empty(unique.size(dim)).scatter_(
        dim, inverse, perm)
    return unique_sorted_index