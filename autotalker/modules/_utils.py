import torch


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