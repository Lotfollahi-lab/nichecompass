"""
This module contains generic VGAE functionalities, added as a Mixin to the
Variational Gene Program Graph Autoencoder module.
"""

import torch


class VGAEModuleMixin:
    """
    VGAE module mix in class containing universal VGAE module 
    functionalities.
    """
    def reparameterize(self,
                       mu: torch.Tensor,
                       logstd: torch.Tensor) -> torch.Tensor:
        """
        Use reparameterization trick for latent space normal distribution.
        
        Parameters
        ----------
        mu:
            Expected values of the latent space distribution.
        logstd:
            Log standard deviations of the latent space distribution.

        Returns
        ----------
        rep:
            Reparameterized latent space values.
        """
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(mu)
            rep = eps.mul(std).add(mu)
            return rep
        else:
            rep = mu
            return rep