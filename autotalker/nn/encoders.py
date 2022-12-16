"""
This module contains encoders used by the Autotalker model.
"""

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv


class GraphEncoder(nn.Module):
    """
    Graph Convolutional Network encoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the input space features x and the edge indices as input, computes one 
    shared GCN layer and one subsequent separate GCN layer to output mu and 
    logstd of the latent space normal distribution respectively.

    Parameters
    ----------
    n_input:
        Number of input nodes to the GCN encoder.
    n_hidden:
        Number of hidden nodes outputted by the first GCN layer.
    n_latent:
        Number of output nodes from the GCN encoder, making up the latent space
        features.
    n_addon_latent:
        Number of add-on nodes in the latent space (new gene programs).
    conv_layer:
        Convolutional layer used.
    n_attention_heads:
        Only relevant if ´conv_layer == gatv2conv´. Number of attention heads
        used.
    dropout_rate:
        Probability of nodes to be dropped in the hidden layer during training.
    activation:
        Activation function used in the first GCN layer.
    """
    def __init__(self,
                 n_input: int,
                 n_hidden: int,
                 n_latent: int,
                 n_addon_latent: int=0,
                 conv_layer: Literal["gcnconv", "gatv2conv"]="gcnconv",
                 n_attention_heads: int=4,
                 dropout_rate: float=0.,
                 activation: nn.Module=nn.ReLU):
        super().__init__()
        self.n_addon_latent = n_addon_latent

        print(f"GRAPH ENCODER -> n_input: {n_input}, n_hidden: {n_hidden}, "
              f"n_latent: {n_latent}, n_addon_latent: {n_addon_latent}, "
              f"conv_layer: {conv_layer}, n_attention_heads: "
              f"{n_attention_heads if conv_layer == 'gatv2conv' else '-'}, "
              f"dropout_rate: {dropout_rate}")

        if conv_layer == "gcnconv":
            self.conv_l1 = GCNConv(n_input, n_hidden)
            self.conv_mu = GCNConv(n_hidden, n_latent)
            self.conv_logstd = GCNConv(n_hidden, n_latent)
            if n_addon_latent != 0:
                self.addon_conv_mu = GCNConv(n_hidden, n_addon_latent)
                self.addon_conv_logstd = GCNConv(n_hidden, n_addon_latent)
        elif conv_layer == "gatv2conv":
            self.conv_l1 = GATv2Conv(n_input,
                                    n_hidden,
                                    heads=n_attention_heads,
                                    concat=False)
            self.conv_mu = GATv2Conv(n_hidden,
                                    n_latent,
                                    heads=n_attention_heads,
                                    concat=False)
            self.conv_logstd = GATv2Conv(n_hidden,
                                        n_latent,
                                        heads=n_attention_heads,
                                        concat=False)
            if n_addon_latent != 0:
                self.addon_conv_mu = GATv2Conv(n_hidden,
                                              n_addon_latent,
                                              heads=n_attention_heads,
                                              concat=False)
                self.addon_conv_logstd = GATv2Conv(n_hidden,
                                                  n_addon_latent,
                                                  heads=n_attention_heads,
                                                  concat=False)

        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GCN encoder.

        Parameters
        ----------
        x:
            Tensor containing the gene expression input features.
        edge_index:
            Tensor containing the edge indices for message passing.
        
        Returns
        ----------
        mu:
            Tensor containing the expected values of the latent space normal 
            distribution.
        logstd:
            Tensor containing the log standard deviations of the latent space
            normal distribution.     
        """
        # Part of forward pass shared across all nodes
        hidden = self.dropout(self.activation(self.conv_l1(x, edge_index)))

        # Part of forward pass only for maskable latent nodes
        mu = self.conv_mu(hidden, edge_index)
        logstd = self.conv_logstd(hidden, edge_index)
        
        # Part of forward pass only for unmaskable add-on latent nodes
        if self.n_addon_latent != 0:
            mu = torch.cat(
                (mu, self.addon_conv_mu(hidden, edge_index)),
                dim=1)
            logstd = torch.cat(
                (logstd, self.addon_conv_logstd(hidden, edge_index)),
                dim=1)
        return mu, logstd