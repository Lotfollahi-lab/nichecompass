import torch
import torch.nn as nn

from autotalker.nn import DotProductDecoder
from autotalker.nn import GCNEncoder
from autotalker.nn import MaskedLinearDecoder
from ._vgaemodulemixin import VGAEModuleMixin
from ._losses import compute_vgae_loss
from ._losses import compute_vgae_loss_parameters
from ._losses import compute_x_recon_loss


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
                 expr_decoder_recon_loss: str,
                 expr_decoder_mask,
                 encoder_dropout_rate: float=0.0,
                 graph_decoder_dropout_rate: float=0.0):
        super().__init__()
        self.encoder = GCNEncoder(
            n_input = encoder_layer_sizes[0],
            n_hidden = encoder_layer_sizes[1],
            n_latent = encoder_layer_sizes[2],
            dropout_rate = encoder_dropout_rate,
            activation = torch.relu)
        
        self.graph_decoder = DotProductDecoder(
            dropout_rate=graph_decoder_dropout_rate)
        self.expr_decoder = MaskedLinearDecoder(
            n_input=expr_decoder_layer_sizes[0],
            n_output=expr_decoder_layer_sizes[1],
            mask=expr_decoder_mask,
            recon_loss=expr_decoder_recon_loss)


    def forward(self, x, edge_index):
        self.mu, self.logstd = self.encoder(x, edge_index)
        self.z = self.reparameterize(self.mu, self.logstd)
        adj_recon_logits = self.graph_decoder(self.z)
        x_recon = self.expr_decoder(self.z)
        return adj_recon_logits, x_recon, self.mu, self.logstd


    def loss(self, adj_recon_logits, x_recon, train_data, mu, logstd, device):
        vgae_loss_norm_factor, vgae_loss_pos_weight = \
        compute_vgae_loss_parameters(train_data.edge_index)

        vgae_loss_pos_weight = vgae_loss_pos_weight.to(device)

        vgae_loss = compute_vgae_loss(
            adj_recon_logits=adj_recon_logits,
            edge_label_index=train_data.edge_index,
            pos_weight=vgae_loss_pos_weight,
            mu=mu,
            logstd=logstd,
            n_nodes=train_data.x.size(0),
            norm_factor=vgae_loss_norm_factor)

        expr_recon_loss = compute_x_recon_loss(x_recon, train_data.x)

        ivgae_loss = vgae_loss + expr_recon_loss

        return ivgae_loss