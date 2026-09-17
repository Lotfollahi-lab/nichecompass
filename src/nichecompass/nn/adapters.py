"""
This module contains adapters that add a small amount of trainable capacity to
an otherwise frozen encoder, for adapting a reference model to query data.
"""

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv


class GraphAdapter(nn.Module):
    """
    Residual bottleneck with its own message passing, for query adaptation.

    Computes ´h + up(conv(act(down(h)), edge_index))´, with ´up´
    initialized to zero so that the module is exactly the identity before it
    is trained.

    Three properties make this the shape of adaptation a spatial query wants,
    none of which the alternatives have together:

    - It performs its own message passing, so unlike anything applied to the
      encoder's input it sees the cell's ACTUAL neighbourhood and can respond
      to which cell types surround it, not only to how many.
    - It starts as the identity and sits alongside a frozen path, so a query
      run departs from the reference gradually rather than from a random
      initialization. A fully unfrozen encoder has no such anchor.
    - It adds parameters instead of changing the shape of existing ones, so it
      can be attached to a reference that has ALREADY been trained. Injecting
      covariate embeddings into the encoder cannot: that changes the encoder's
      input dimension.

    It is deliberately placed before the frozen graph convolutions that
    produce ´mu´, and never on ´mu´ itself. ´mu´ IS the gene program
    activities, so transforming it would move the axes the gene program
    loadings define, which is the one thing freezing the decoder exists to
    prevent.

    Parameters
    ----------
    n_input:
        Dimensionality of the representation being adapted.
    n_bottleneck:
        Width of the bottleneck. Small relative to ´n_input´ is the point: it
        bounds how far the query can depart from the reference.
    conv_layer:
        Graph convolution used inside the bottleneck.
    n_attention_heads:
        Number of attention heads, used only for ´"gatv2conv"´.
    activation:
        Activation applied after the down projection.
    """
    def __init__(self,
                 n_input: int,
                 n_bottleneck: int,
                 conv_layer: Literal["gcnconv", "gatv2conv"]="gcnconv",
                 n_attention_heads: int=4,
                 activation=torch.relu):
        super().__init__()
        if n_bottleneck <= 0:
            raise ValueError("´n_bottleneck´ must be a positive integer.")
        self.activation = activation
        self.down = nn.Linear(n_input, n_bottleneck)
        if conv_layer == "gcnconv":
            self.conv = GCNConv(n_bottleneck, n_bottleneck)
        elif conv_layer == "gatv2conv":
            self.conv = GATv2Conv(n_bottleneck,
                                  n_bottleneck,
                                  heads=n_attention_heads,
                                  concat=False)
        else:
            raise ValueError(
                f"´conv_layer´ is {conv_layer!r}, which is neither 'gcnconv' "
                "nor 'gatv2conv'.")
        self.up = nn.Linear(n_bottleneck, n_input)
        # Exact identity at initialization. Without this the adapter would
        # inject noise into a frozen encoder on the first forward, which is
        # the opposite of anchoring to the reference.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        print(f"GRAPH ADAPTER -> n_input: {n_input}, "
              f"n_bottleneck: {n_bottleneck}, conv_layer: {conv_layer}")

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Adapt a node representation using its neighbourhood.

        Parameters
        ----------
        x:
            Node representation (dim: n_obs x n_input).
        edge_index:
            Graph connectivity (dim: 2 x n_edges).

        Returns
        ----------
        x:
            Adapted node representation, equal to the input until trained.
        """
        # One nonlinearity, as in the standard adapter bottleneck. A second
        # one after the convolution zeroes out too much at small bottleneck
        # widths: with two ReLUs a node whose projection is entirely negative
        # receives no adaptation at all.
        hidden = self.activation(self.down(x))
        hidden = self.conv(hidden, edge_index)
        return x + self.up(hidden)
