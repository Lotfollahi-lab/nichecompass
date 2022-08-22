import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the input space features x and the edge indices as input, computes one 
    shared GCN layer and one subsequent separate GCN layer to output mu and 
    logstd of the latent space distribution respectively.

    Parameters
    ----------
    n_input:
        Number of input nodes to the GCN encoder.
    n_hidden:
        Number of hidden nodes outputted by the first GCN layer.
    n_latent:
        Number of output nodes from the GCN encoder, making up the latent space
        features.
    dropout_rate:
        Probability of nodes to be dropped in the hidden layer during training.
    activation:
        Activation function used in the first GCN layer.
    """
    def __init__(self,
                 n_input: int,
                 n_hidden: int,
                 n_latent: int,
                 dropout_rate: float=0.0,
                 activation: nn.Module=nn.ReLU):
        super().__init__()

        print(f"GCN ENCODER - n_input: {n_input}, n_hidden: {n_hidden}, n_latent"
              f": {n_latent}, dropout_rate: {dropout_rate}")

        self.gcn_l1 = GCNConv(n_input, n_hidden)
        self.gcn_mu = GCNConv(n_hidden, n_latent)
        self.gcn_logstd = GCNConv(n_hidden, n_latent)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x, edge_index):
        hidden = self.dropout(self.activation(self.gcn_l1(x, edge_index)))
        mu = self.gcn_mu(hidden, edge_index)
        logstd = self.gcn_logstd(hidden, edge_index)
        return mu, logstd