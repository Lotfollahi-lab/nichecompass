from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch_geometric

#FIXME represent this as a sequence of modules
#FIXME need to handle injection of covariates in the input by concatenation

Encoder = torch_geometric.nn.Sequ, [
    (torch.nn.Linear(dataset.num_features, 1000), "x -> x"),
    torch_geometric.nn.Sequential("x, edge_index -> x_mean",
        (torch_geometric.nn.GATv2Conv(1000, 100, 4, concat=False), "x, edge_index -> x_mean"),
        torch.nn.ReLU(),
        torch.nn.Dropout(0)
    ),
    torch_geometric.nn.Sequential("x, edge_index -> x_logstd",
        (torch_geometric.nn.GATv2Conv(1000, 100, 4, concat=False), "x, edge_index -> x_logstd"),
        torch.nn.ReLU(),
        torch.nn.Dropout(0)
    ),
    (lambda: x_mean, x_logstd: (x_mean, x_logstd), "x_mean, x_std -> x_latent")
])





















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
                 n_addon_latent: int=10,
                 n_fc_layers: int=1,
                 conv_layer: Literal["gcnconv", "gatv2conv"]="gcnconv",
                 n_layers: int=1,
                 cat_covariates_embed_mode: Literal["input", "hidden"]="input",
                 n_attention_heads: int=4,
                 dropout_rate: float=0.,
                 activation: nn.Module=nn.ReLU,
                 use_bn: bool=True):
        super().__init__()
        self.n_addon_latent = n_addon_latent
        self.n_layers = n_layers
        self.n_fc_layers = n_fc_layers
        self.cat_covariates_embed_mode = cat_covariates_embed_mode
        self.use_bn = use_bn

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
                self.conv_l1 = torch_geometric.nn.GCNConv(n_hidden,
                                       n_hidden)
            self.conv_mu = torch_geometric.nn.GCNConv(n_hidden,
                                   n_latent)
            self.conv_logstd = torch_geometric.nn.GCNConv(n_hidden,
                                       n_latent)
            if n_addon_latent != 0:
                self.addon_conv_mu = torch_geometric.nn.GCNConv(n_hidden,
                                             n_addon_latent)
                self.addon_conv_logstd = torch_geometric.nn.GCNConv(n_hidden,
                                                 n_addon_latent)
        elif conv_layer == "gatv2conv":
            if n_layers == 2:
                self.conv_l1 = torch_geometric.nn.GATv2Conv(n_hidden,
                                         n_hidden,
                                         heads=n_attention_heads,
                                         concat=False)
            self.conv_mu = torch_geometric.nn.GATv2Conv(n_hidden,
                                     n_latent,
                                     heads=n_attention_heads,
                                     concat=False)
            self.conv_logstd = torch_geometric.nn.GATv2Conv(n_hidden,
                                         n_latent,
                                         heads=n_attention_heads,
                                         concat=False)
            if n_addon_latent != 0:
                self.addon_conv_mu = torch_geometric.nn.GATv2Conv(n_hidden,
                                               n_addon_latent,
                                               heads=n_attention_heads,
                                               concat=False)
                self.addon_conv_logstd = torch_geometric.nn.GATv2Conv(n_hidden,
                                                   n_addon_latent,
                                                   heads=n_attention_heads,
                                                   concat=False)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        if use_bn:
            self.bn_mu = nn.BatchNorm1d(n_hidden, affine=True)
        #self.final_activation = nn.Tanh()

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
            if cat_covariates_embed is not None:
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
        if self.use_bn:
            mu = self.bn_mu(mu)
        #mu = self.final_activation(mu)
        return mu, logstd
