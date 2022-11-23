"""
This module contains generic VGAE functionalities, added as a Mixin to the main
Variational Gene Program Graph Autoencoder module.
"""

from typing import Tuple

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

    @torch.no_grad()
    def get_latent_representation(self,
                                  x: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  return_mu_std: bool=False) -> torch.Tensor:
        """
        Encode input features x and edge index into the latent space normal 
        distribution parameters and return z. If the module is not in training
        mode, mu will be returned.
           
        Parameters
        ----------
        x:
            Feature matrix to be encoded into latent space.
        edge_index:
            Edge index of the graph.
        return_mu_std:
            If `True`, return mu and logstd instead of a random sample from the
            latent space.

        Returns
        -------
        z:
            Latent space encoding.
        """
        mu, logstd = self.encoder(x, edge_index)
        if return_mu_std:
            return mu, torch.exp(logstd)
        else:
            z = self.reparameterize(mu, logstd)
            return z

    @torch.no_grad()
    def get_zinb_gene_expr_params(self,
                                  z: torch.Tensor,
                                  log_library_size: torch.Tensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent space features z using the log library size of the
        input gene expression to return the parameters of the ZINB distribution
        used for reconstruction of gene expression.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.
        log_library_size:
            Tensor containing the log library size of the nodes.

        Returns
        ----------
        zinb_parameters:
            Parameters for the ZINB distribution to model gene expression.
        """
        zinb_parameters = self.gene_expr_decoder(
            z,
            log_library_size)
        return zinb_parameters
      