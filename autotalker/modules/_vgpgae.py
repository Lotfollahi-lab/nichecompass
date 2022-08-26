import torch
import torch.nn as nn

from autotalker.nn import GCNEncoder
from autotalker.nn import DotProductGraphDecoder
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
                 encoder_layer_sizes: list,
                 expr_decoder_layer_sizes: list,
                 expr_decoder_mask: torch.Tensor,
                 dropout_rate: float=0.0):
        super().__init__()
        self.n_input = encoder_layer_sizes[0]
        self.n_hidden = encoder_layer_sizes[1]
        self.n_latent = encoder_layer_sizes[2]
        self.dropout_rate = dropout_rate

        print("--- INITIALIZING NEW NETWORK MODULE: VGPGAE ---")

        self.encoder = GCNEncoder(n_input = encoder_layer_sizes[0],
                                  n_hidden = encoder_layer_sizes[1],
                                  n_latent = encoder_layer_sizes[2],
                                  dropout_rate = dropout_rate,
                                  activation = torch.relu)
        
        self.graph_decoder = DotProductGraphDecoder(dropout_rate=dropout_rate)

        self.expr_decoder = MaskedGeneExprDecoder(
            n_input=expr_decoder_layer_sizes[0],
            n_output=expr_decoder_layer_sizes[1],
            mask=expr_decoder_mask)
        
        # Gene-specific inverse dispersion parameters
        self.theta = torch.nn.Parameter(torch.randn(self.n_input))

    def forward(self, x, edge_index):
        log_library_size = torch.log(x.sum(1)).unsqueeze(1)
        self.mu, self.logstd = self.encoder(x, edge_index)
        self.z = self.reparameterize(self.mu, self.logstd)
        adj_recon_logits = self.graph_decoder(self.z)
        zinb_parameters = self.expr_decoder(self.z)
        outputs = (adj_recon_logits, zinb_parameters, log_library_size, self.mu,
                   self.logstd)
        return outputs

    def loss(self,
             adj_recon_logits,
             zinb_parameters,
             data_batch,
             mu,
             logstd,
             device):
        vgae_loss_params = vgae_loss_parameters(data_batch=data_batch,
                                                device=device)
        edge_recon_loss_norm_factor = vgae_loss_params[0]
        edge_recon_loss_pos_weight = vgae_loss_params[1]

        vgae_loss = compute_vgae_loss(
            adj_recon_logits=adj_recon_logits,
            edge_label_index=data_batch.edge_label_index,
            edge_labels=data_batch.edge_label,
            edge_recon_loss_pos_weight=edge_recon_loss_pos_weight,
            edge_recon_loss_norm_factor=edge_recon_loss_norm_factor,
            mu=mu,
            logstd=logstd,
            n_nodes=data_batch.x.size(0))

        nb_mean, zi_prob_logits = zinb_parameters

        # Inverse dispersion
        theta = torch.exp(self.theta)

        gene_expr_recon_loss = compute_gene_expr_recon_zinb_loss(
            x=data_batch.x,
            mu=nb_mean,
            theta=theta,
            zi_prob_logits=zi_prob_logits)

        vgpgae_loss = vgae_loss + gene_expr_recon_loss
        return vgpgae_loss, vgae_loss, gene_expr_recon_loss