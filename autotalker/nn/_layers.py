import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer class as per Kipf, T. N. & Welling, M.
    Semi-Supervised Classification with Graph Convolutional Networks. arXiv
    [cs.LG] (2016).

    Parameters
    ----------
    n_input:
        Number of input nodes to the GCN Layer.
    n_output:
        Number of output nodes from the GCN layer.
    dropout:
        Probability of nodes to be dropped during training.
    activation:
        Activation function used in the GCN layer.
    """
    def __init__(
            self,
            n_input: int,
            n_output: int,
            dropout_rate: float=0.0,
            activation=torch.relu):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation
        self.weights = nn.Parameter(torch.FloatTensor(n_input, n_output))
        self.initialize_weights()

    def initialize_weights(self):
        # Glorot weight initialization
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, input, adj):
        output = self.dropout(input)
        output = torch.mm(output, self.weights)
        output = torch.sparse.mm(adj, output)
        return self.activation(output)


class FCLayer(nn.Module):
    """
    Fully connected layer class.

    Parameters
    ----------
    n_input:
        Number of input nodes to the FC Layer.
    n_output:
        Number of output nodes from the FC layer.
    activation:
        Activation function used in the FC layer.
    """
    def __init__(self, n_input: int, n_output: int, activation=nn.ReLU):
        super(FCLayer, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, input: torch.Tensor):
        output = self.linear(input)
        if self.activation is not None:
            return self.activation(output)
        else:
            return output


class MaskedFCLayer(nn.Linear):
    """
    Adapted from https://github.com/theislab/scarches. 
    """
    def __init__(self, n_input: int, n_output: int, mask, bias=True):
        # Mask should have dim n_input x n_output
        if n_input != mask.shape[0] or n_output != mask.shape[1]:
            raise ValueError("Incorrect shape of the mask. Mask should have dim"
                             "n_input x n_output")
        super().__init__(n_input, n_output, bias)
        
        self.register_buffer("mask", mask.t())

        # Zero out weights with the mask
        # Gradient descent does not consider these zero weights
        self.weight.data *= self.mask

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)
