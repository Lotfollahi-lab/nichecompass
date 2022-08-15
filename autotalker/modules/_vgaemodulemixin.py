import logging
from typing import Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


class VGAEModuleMixin:
    """Universal VGAE module functionalities."""


    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(mu)
            return eps.mul(std).add(mu)
        else:
            return mu


    @torch.no_grad()
    def get_latent_representation(self, x, edge_index):
        """
        Map input x and edge index into the latent space z and return z.
           
        Parameters
        ----------
        x:
            Feature matrix to be mapped into latent space.
        edge_index:
            Corresponding edge index of the graph.

        Returns
        -------
        z:
            Tensor containing latent space encoding z.
        """
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return z

