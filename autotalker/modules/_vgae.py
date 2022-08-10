# -*- coding: utf-8 -*-
"""Main module."""


import torch
import torch.nn as nn
from scvi.module.base import BaseModuleClass

from ..nn._base_components import DotProductDecoder
from ..nn._base_components import GCNEncoder
from ..nn._base_components import FCLayer

torch.backends.cudnn.benchmark = True


class VGAE(BaseModuleClass):
    """
    Variational Graph Autoencoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Parameters
    ----------
    n_input
        Number of nodes in the input layer.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Number of nodes in the latent space.
    dropout
        Probability that nodes will be dropped during training.
    """
    def __init__(
            self,
            n_input: int,
            n_hidden: int,
            n_latent: int,
            dropout_rate: float=0.0):
        super().__init__()
        self.encoder = GCNEncoder(
            n_input = n_input,
            n_hidden = n_hidden,
            n_latent = n_latent,
            dropout_rate = dropout_rate,
            activation = torch.relu)
        
        self.decoder = DotProductDecoder(dropout_rate = dropout_rate)

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(mu)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, X, A):
        self.mu, self.logstd = self.encoder(X, A)
        self.Z = self.reparameterize(self.mu, self.logstd)
        A_rec_logits = self.decoder(self.Z)
        return A_rec_logits, self.mu, self.logstd