from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from autotalker.nn import DotProductGraphDecoder
from autotalker.nn import GCNEncoder
from autotalker.nn import MaskedLinearExprDecoder
from ._vgaemodulemixin import VGAEModuleMixin
from ._losses import compute_vgae_loss

class IVGAE(nn.Module, VGAEModuleMixin):
    """
    Interpretable Variational Graph Autoencoder class.

    Parameters
    ----------
    n_input:
        Number of nodes in the input layer.
    n_hidden:
        Number of nodes in the hidden layer.
    n_latent:
        Number of nodes in the latent space.
    dropout_rate:
        Probability that nodes will be dropped during training.
    """
    def __init__(self,
                 encoder_layer_sizes: list,
                 expr_decoder_layer_sizes: list,
                 expr_decoder_mask,
                 conditions: list,
                 encoder_dropout_rate: float=0.0,
                 graph_decoder_dropout_rate: float=0.0,
                 expr_decoder_recon_loss: Literal["nb", "mse"]="nb"):
        super().__init__()

        print("INITIALIZING NEW NETWORK MODULE..........")

        self.n_input = encoder_layer_sizes[0]
        self.n_hidden = encoder_layer_sizes[1]
        self.n_latent = encoder_layer_sizes[2]
        self.n_condition = len(conditions)
        self.conditions = conditions
        self.condition_label_dict = {k: v for k, v in zip(conditions, range(len(conditions)))}
        self.dropout_rate = encoder_dropout_rate
        self.encoder = GCNEncoder(
            n_input = encoder_layer_sizes[0],
            n_hidden = encoder_layer_sizes[1],
            n_latent = encoder_layer_sizes[2],
            dropout_rate = encoder_dropout_rate,
            activation = torch.relu)
        
        self.graph_decoder = DotProductGraphDecoder(
            dropout_rate=graph_decoder_dropout_rate)
        self.expr_decoder = MaskedLinearExprDecoder(
            n_input=expr_decoder_layer_sizes[0],
            n_output=expr_decoder_layer_sizes[1],
            mask=expr_decoder_mask,
            n_condition=self.n_condition,
            recon_loss=expr_decoder_recon_loss)
        self.expr_decoder_recon_loss = expr_decoder_recon_loss
        
        if expr_decoder_recon_loss in ["nb"]:
            self.theta = torch.nn.Parameter(torch.randn(self.n_input, self.n_condition))
        else:
            self.theta = None


    def forward(self, x, edge_index):
        self.mu, self.logstd = self.encoder(x, edge_index)
        self.z = self.reparameterize(self.mu, self.logstd)
        adj_recon_logits = self.graph_decoder(self.z)
        expr_decoder_output = self.expr_decoder(self.z)
        return adj_recon_logits, expr_decoder_output, self.mu, self.logstd


    def loss(self, adj_recon_logits, expr_decoder_output, data_batch, mu, logstd, device):
        vgae_loss_norm_factor, vgae_loss_pos_weight = \
        compute_vgae_loss_parameters(data_batch.edge_index)

        vgae_loss_pos_weight = vgae_loss_pos_weight.to(device)

        vgae_loss = compute_vgae_loss(
            adj_recon_logits=adj_recon_logits,
            edge_label_index=data_batch.edge_label_index,
            pos_weight=vgae_loss_pos_weight,
            mu=mu,
            logstd=logstd,
            n_nodes=data_batch.x.size(0),
            norm_factor=vgae_loss_norm_factor)

        if self.expr_decoder_recon_loss == "nb": ### FIX ####
            ### Log FORWARD PASS ??? ###
            dec_mean_gamma = expr_decoder_output
            size_factors = data_batch.size_factors.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dispersion = torch.exp(F.linear(self.theta))
            dec_mean = dec_mean_gamma * size_factors
            expr_recon_loss = compute_x_recon_nb_loss(x=data_batch.x,
                                                      mu=dec_mean,
                                                      theta=dispersion)
        elif self.expr_decoder_recon_loss == "mse":
            x_recon = expr_decoder_output
            expr_recon_loss = compute_x_recon_mse_loss(x_recon=x_recon,
                                                       x=data_batch.x)

        ivgae_loss = vgae_loss + expr_recon_loss

        return ivgae_loss, vgae_loss, expr_recon_loss