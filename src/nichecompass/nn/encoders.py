"""
This module contains the encoder used by the NicheCompass model.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv


class Encoder(nn.Module):
    """
    Encoder class.

    Takes the input space features x and the edge indices as input, first computes
    fully connected layers and then uses message passing layers to output mu and 
    logstd of the latent space normal distribution.

    Parameters
    ----------
    n_input:
        Number of input nodes (omics features) to the encoder.
    n_cat_covariates_embed_input:
        Number of categorical covariates embedding input nodes to the encoder.
    n_hidden:
        Number of hidden nodes outputted after the fully connected layers and
        intermediate message passing layers.
    n_latent:
        Number of output nodes (prior gps) from the encoder, making up the
        first part of the latent space features z.
    n_addon_latent:
        Number of add-on nodes in the latent space (new gps), making up the
        second part of the latent space features z.
    n_fc_layers:
        Number of fully connected layers before the message passing layers.
    conv_layer:
        Message passing layer used.
    n_layers:
        Number of message passing layers.
    cat_covariates_embed_mode:
        Indicates where to inject the categorical covariates embedding if
        injected.
    n_attention_heads:
        Only relevant if ´conv_layer == gatv2conv´. Number of attention heads
        used.
    dropout_rate:
        Probability of nodes to be dropped in the hidden layer during training.
    activation:
        Activation function used after the fully connected layers and
        intermediate message passing layers.
    use_bn:
        If ´True´, use a batch normalization layer at the end to normalize ´mu´.
    """
    def __init__(self,
                 n_input: int,
                 n_cat_covariates_embed_input: int,
                 n_hidden: int,
                 n_latent: int,
                 n_addon_latent: int=100,
                 n_fc_layers: int=1,
                 conv_layer: Literal["gcnconv", "gatv2conv"]="gatv2conv",
                 n_layers: int=1,
                 cat_covariates_embed_mode: Literal["input", "hidden"]="input",
                 n_attention_heads: int=4,
                 dropout_rate: float=0.,
                 activation: nn.Module=nn.ReLU,
                 use_bn: bool=True):
        super().__init__()
        print("ENCODER -> "
              f"n_input: {n_input}, "
              f"n_cat_covariates_embed_input: {n_cat_covariates_embed_input}, "
              f"n_hidden: {n_hidden}, "
              f"n_latent: {n_latent}, "
              f"n_addon_latent: {n_addon_latent}, "
              f"n_fc_layers: {n_fc_layers}, "
              f"n_layers: {n_layers}, "
              f"conv_layer: {conv_layer}, "
              f"n_attention_heads: "
              f"{n_attention_heads if conv_layer == 'gatv2conv' else '0'}, "
              f"dropout_rate: {dropout_rate}, ")

        self.n_addon_latent = n_addon_latent
        self.n_layers = n_layers
        self.n_fc_layers = n_fc_layers
        self.cat_covariates_embed_mode = cat_covariates_embed_mode
        
        if ((cat_covariates_embed_mode == "input") &
            (n_cat_covariates_embed_input != 0)):
            # Add categorical covariates embedding to input
            n_input += n_cat_covariates_embed_input
        
        if n_fc_layers == 2:
            self.fc_l1 = nn.Linear(n_input, int(n_input / 2))
            self.fc_l2 = nn.Linear(int(n_input / 2), n_hidden)
            self.fc_l2_bn = nn.BatchNorm1d(n_hidden)
        elif n_fc_layers == 1:
            self.fc_l1 = nn.Linear(n_input, n_hidden)
        
        if ((cat_covariates_embed_mode == "hidden") &
            (n_cat_covariates_embed_input != 0)):
            # Add categorical covariates embedding to hidden after fc_l
            n_hidden += n_cat_covariates_embed_input

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
                cat_covariates_embed: Optional[torch.Tensor]=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x:
            Tensor containing the omics features.
        edge_index:
            Tensor containing the edge indices for message passing.
        cat_covariates_embed:
            Tensor containing the categorical covariates embedding (all
            categorical covariates embeddings concatenated into one embedding).
        
        Returns
        ----------
        mu:
            Tensor containing the expected values of the latent space normal 
            distribution.
        logstd:
            Tensor containing the log standard deviations of the latent space
            normal distribution.
        """
        if ((self.cat_covariates_embed_mode == "input") &
            (cat_covariates_embed is not None)):
            # Add categorical covariates embedding to input vector
            x = torch.cat((x,
                           cat_covariates_embed),
                          axis=1)
        
        # FC forward pass shared across all nodes
        hidden = self.dropout(self.activation(self.fc_l1(x)))
        if self.n_fc_layers == 2:
            hidden = self.dropout(self.activation(self.fc_l2(hidden)))
            hidden = self.fc_l2_bn(hidden)
        
        if ((self.cat_covariates_embed_mode == "hidden") &
            (cat_covariates_embed is not None)):
            # Add categorical covariates embedding to hidden vector
            hidden = torch.cat((hidden,
                                cat_covariates_embed),
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
    