"""
This module contains helper functions for the ´nn´ subpackage.
"""

import torch


def compute_cosine_similarity(tensor1: torch.Tensor,
                              tensor2: torch.Tensor,
                              eps: float=1e-8) -> torch.Tensor:
    """
    Compute the cosine similarity between two tensors.

    Parameters
    ----------
    tensor1:
        First tensor for cosine similarity computation (dim: n_obs x n_features).
    tensor2:
        Second tensor for cosine similarity computation (dim: n_obs x
        n_features).
    
    Returns
    ----------
    cosine_sim:
        Tensor that contains the computed cosine similarities (dim: n_obs x 
        n_obs).
    """
    tensor1_norm = tensor1.norm(dim=1)[:, None]
    tensor2_norm = tensor2.norm(dim=1)[:, None]
    tensor1_normalized = tensor1 / torch.max(
            tensor1_norm, eps * torch.ones_like(tensor1_norm))
    tensor2_normalized = tensor2 / torch.max(
            tensor2_norm, eps * torch.ones_like(tensor2_norm))
    cosine_sim = torch.mm(tensor1_normalized,
                          tensor2_normalized.transpose(0, 1))
    return cosine_sim