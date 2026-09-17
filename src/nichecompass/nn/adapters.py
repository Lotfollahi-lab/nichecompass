"""
This module contains adapters that add a small amount of trainable capacity to
an otherwise frozen encoder, for adapting a reference model to query data.
"""

import torch
import torch.nn as nn


class GraphAdapter(nn.Module):
    """
    Residual bottleneck that adapts the input of an existing graph convolution.

    Computes ´h + up(act(down(h)))´, with ´up´ initialized to zero so that the
    module is exactly the identity before it is trained.

    It performs NO message passing of its own. That is the point: the
    correction it adds is per cell, and the graph convolution that already
    follows it is what mixes those corrections across each cell's
    neighbourhood. Concretely, the convolution then computes

        mu_i = conv({h_j + delta(h_j) : j in N(i) and i})

    and because ´delta´ is a function of each cell's own representation, the
    aggregated correction depends on WHICH cells are neighbours, not only on
    how many. The adapter is therefore neighbourhood composition sensitive
    even though it is a per cell transform: the existing convolution supplies
    the graph, the adapter supplies the trainable part.

    Giving the adapter its own convolution instead would be a mistake, and was
    the first version of this module. It adds a hop, so a single layer encoder
    aggregating over one hop becomes two and the query's gene program
    activities would summarize a larger spatial region than the reference's -
    exactly the comparability the freeze exists to protect. It also breaks the
    minibatch: the loaders sample ´loaders_n_hops´ hops, one by default, so a
    second aggregation reads neighbours whose own neighbourhoods the sampler
    truncated, and silently computes the wrong value for every seed node.

    Adapters are placed in front of the graph convolutions and never on ´mu´.
    ´mu´ IS the gene program activities; adapting the map that produces it is
    legitimate, and is what unfreezing the encoder does, but transforming
    ´mu´ after the fact would move the axes the frozen loadings define.

    Parameters
    ----------
    n_input:
        Dimensionality of the representation being adapted.
    n_bottleneck:
        Width of the bottleneck. Small relative to ´n_input´ is the point: it
        bounds how far the query can depart from the reference.
    activation:
        Activation applied after the down projection.
    """
    def __init__(self,
                 n_input: int,
                 n_bottleneck: int,
                 activation=torch.relu):
        super().__init__()
        if n_bottleneck <= 0:
            raise ValueError("´n_bottleneck´ must be a positive integer.")
        self.activation = activation
        self.down = nn.Linear(n_input, n_bottleneck)
        self.up = nn.Linear(n_bottleneck, n_input)
        # Exact identity at initialization. Without this the adapter would
        # inject noise into a frozen encoder on the first forward, which is
        # the opposite of anchoring to the reference.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        print(f"GRAPH ADAPTER -> n_input: {n_input}, "
              f"n_bottleneck: {n_bottleneck}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add a bounded per cell correction to a node representation.

        Parameters
        ----------
        x:
            Node representation (dim: n_obs x n_input).

        Returns
        ----------
        x:
            Adapted node representation, equal to the input until trained.
            The graph convolution that follows turns these per cell
            corrections into a neighbourhood dependent one.
        """
        return x + self.up(self.activation(self.down(x)))
