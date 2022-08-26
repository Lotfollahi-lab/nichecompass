from typing import Tuple

import torch
import torch.nn as nn
import torch_geometric

from autotalker.nn import DotProductGraphDecoder
from autotalker.nn import GCNEncoder
from autotalker.nn import MaskedGeneExprDecoder
from ._vgaemodulemixin import VGAEModuleMixin
from ._losses import compute_gene_expr_recon_zinb_loss
from ._losses import compute_vgae_loss
from ._losses import vgae_loss_parameters


class VGPGAE(nn.Module, VGAEModuleMixin):
    """
    Variational Gene Program Graph Autoencoder class.

    Parameters
    ----------
    n_input:
        Number of nodes in the input layer.
    n_hidden:
        Number of nodes in the hidden layer.
    n_latent:
        Number of nodes in the latent space.
    use_size_factor_key:
        If `True` use size factors under key. If `False` use observed lib size.
    dropout_rate:
        Probability that nodes will be dropped during training.
    """
    def __init__(self,
                 n_input: int,
                 n_hidden: int,
                 n_latent: int,
                 gene_expr_decoder_mask: torch.Tensor,
                 dropout_rate: float=0.0):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate

        print("--- INITIALIZING NEW NETWORK MODULE: VGPGAE ---")

        self.encoder = GCNEncoder(n_input=n_input,
                                  n_hidden=n_hidden,
                                  n_latent=n_latent,
                                  dropout_rate=dropout_rate,
                                  activation=torch.relu)
        
        self.graph_decoder = DotProductGraphDecoder(dropout_rate=dropout_rate)

        self.gene_expr_decoder = MaskedGeneExprDecoder(
            n_input=n_latent,
            n_output=n_input,
            mask=gene_expr_decoder_mask)
        
        # Gene-specific inverse dispersion parameters
        self.theta = torch.nn.Parameter(torch.randn(self.n_input))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        log_library_size = torch.log(x.sum(1)).unsqueeze(1)
        self.mu, self.logstd = self.encoder(x, edge_index)
        self.z = self.reparameterize(self.mu, self.logstd)
        adj_recon_logits = self.graph_decoder(self.z)
        zinb_parameters = self.gene_expr_decoder(self.z, log_library_size)
        output = dict(adj_recon_logits=adj_recon_logits,
                      zinb_parameters=zinb_parameters,
                      mu=self.mu,
                      logstd=self.logstd)
        return output

    def loss(self,
             model_output: dict,
             data_batch: torch_geometric.data.Data,
             device: str):
        vgae_loss_params = vgae_loss_parameters(data_batch=data_batch,
                                                device=device)
        edge_recon_loss_norm_factor = vgae_loss_params[0]
        edge_recon_loss_pos_weight = vgae_loss_params[1]

        vgae_loss = compute_vgae_loss(
            adj_recon_logits=model_output["adj_recon_logits"],
            edge_label_index=data_batch.edge_label_index,
            edge_labels=data_batch.edge_label,
            edge_recon_loss_pos_weight=edge_recon_loss_pos_weight,
            edge_recon_loss_norm_factor=edge_recon_loss_norm_factor,
            mu=model_output["mu"],
            logstd=model_output["logstd"],
            n_nodes=data_batch.x.size(0))

        nb_means, zi_prob_logits = model_output["zinb_parameters"]

        # Gene-specific inverse dispersion
        theta = torch.exp(self.theta)

        gene_expr_recon_loss = compute_gene_expr_recon_zinb_loss(
            x=data_batch.x,
            mu=nb_means,
            theta=theta,
            zi_prob_logits=zi_prob_logits)

        loss = vgae_loss + gene_expr_recon_loss
        return loss