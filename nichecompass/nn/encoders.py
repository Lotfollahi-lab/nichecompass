"""
This module contains encoders used by the NicheCompass model.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv


class Encoder(nn.Module):
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
    n_cond_embed_input:
        Number of conditional embedding input nodes to the GCN encoder.
    n_layers:
        Number of layers.
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
                 n_cond_embed_input: int,
                 n_layers: int,
                 n_hidden: int,
                 n_latent: int,
                 n_addon_latent: int=0,
                 conv_layer: Literal["gcnconv", "gatv2conv"]="gcnconv",
                 cond_embed_mode: Literal["input", "hidden"]="hidden",
                 n_attention_heads: int=4,
                 dropout_rate: float=0.,
                 activation: nn.Module=nn.ReLU):
        super().__init__()
        self.n_addon_latent = n_addon_latent
        self.n_layers = n_layers
        self.cond_embed_mode = cond_embed_mode

        print(f"ENCODER -> n_input: {n_input}, n_cond_embed_input: "
              f"{n_cond_embed_input}, n_layers: {n_layers}, n_hidden: "
              f"{n_hidden}, n_latent: {n_latent}, n_addon_latent: "
              f"{n_addon_latent}, conv_layer: {conv_layer}, n_attention_heads: "
              f"{n_attention_heads if conv_layer == 'gatv2conv' else '0'}, "
              f"dropout_rate: {dropout_rate}")
        
        if (cond_embed_mode == "input") & (n_cond_embed_input != 0):
            # Add conditional embedding to input
            n_input += n_cond_embed_input
        
        self.fc_l = nn.Linear(n_input, n_hidden)
        
        if (cond_embed_mode == "hidden") & (n_cond_embed_input != 0):
            # Add conditional embedding to hidden after fc_l
            n_hidden += n_cond_embed_input

        if conv_layer == "gcnconv":
            if n_layers == 2:
                self.conv_l1 = GCNConv(n_hidden,
                                       n_hidden)
            self.conv_mu = GCNConv(n_hidden,
                                   n_latent)
            self.conv_logstd = GCNConv(n_hidden,
                                       n_latent)
            if n_addon_latent != 0:
                self.addon_conv_mu = GCNConv(n_hidden,
                                             n_addon_latent)
                self.addon_conv_logstd = GCNConv(n_hidden,
                                                 n_addon_latent)           
        elif conv_layer == "gatv2conv":
            if n_layers == 2:
                self.conv_l1 = GATv2Conv(n_hidden,
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
                edge_index: torch.Tensor,
                cond_embed: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass of the GCN encoder.

        Parameters
        ----------
        x:
            Tensor containing the gene expression input features.
        edge_index:
            Tensor containing the edge indices for message passing.
        cond_embed:
            Tensor containing the conditional embedding.
        
        Returns
        ----------
        mu:
            Tensor containing the expected values of the latent space normal 
            distribution.
        logstd:
            Tensor containing the log standard deviations of the latent space
            normal distribution.
        """
        if (self.cond_embed_mode == "input") & (cond_embed is not None):
            # Add conditional embedding to input vector
            if cond_embed is not None:
                x = torch.cat((x,
                               cond_embed),
                              axis=1)
        
        # FC forward pass shared across all nodes
        hidden = self.dropout(self.activation(self.fc_l(x)))
        
        if (self.cond_embed_mode == "hidden") & (cond_embed is not None):
            # Add conditional embedding to hidden vector
            hidden = torch.cat((hidden,
                                cond_embed),
                               axis=1)
        
        if self.n_layers == 2:
            # Part of forward pass shared across all nodes
            hidden = self.dropout(self.activation(
                self.conv_l1(hidden, edge_index)))

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