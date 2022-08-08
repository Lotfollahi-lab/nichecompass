class Discriminator(nn.Module):
    """
    ### WIP ###
    Discriminator class. Adversarial network that takes a sample from the latent
    space as input and uses fully connected layers with a final sigmoid
    activation in the last layer to judge whether a sample is from a prior
    Gaussian distribution or from VGAE.

    Parameters
    ----------
    n_input
        Number of nodes in the input layer.
    n_hidden_l1
        Number of nodes in the first hidden layer.
    n_hidden_l2
        Number of nodes in the second hidden layer.
    activation
        Activation function used in the first and second hidden layer.
    """
    def __init__(
        self,
        n_input: int = 125,
        n_hidden_l1: int = 150,
        n_hidden_l2: int = 125,
        activation=F.relu,
    ):
        super(Discriminator, self).__init__()
        self.fc_l1 = FCLayer(n_input, n_hidden_l1, activation)
        self.fc_l2 = FCLayer(n_hidden_l1, n_hidden_l2, activation)
        self.fc_l3 = FCLayer(n_hidden_l2, 1, None)

    def forward(self, Z):
        hidden = self.fc_l1(Z)
        hidden = self.fc_l2(hidden)
        Y = self.fc_l3(hidden)
        return Y