import torch


def unique_sorted_index(x, dim=-1):
    """
    Utility function to remove duplicates from a tensor and return a sorted 
    index containing only indeces from unique values.
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)