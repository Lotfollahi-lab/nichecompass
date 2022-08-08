class Deeplinc(nn.Module):
    """
    ### WIP ###
    Deeplinc model class as per Li, R. & Yang, X. De novo reconstruction of cell
    interaction landscapes from single-cell spatial transcriptome data with 
    DeepLinc. Genome Biol. 23, 124 (2022).

    Parameters
    ----------
    n_input
        Number of nodes in the input layer.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Number of nodes in the latent space.
    dropout
        Probability that nodes will be dropped during training.
    """
    def __init__(self,
                 adata: ad.AnnData,
                 n_input: int,
                 n_hidden: int,
                 n_latent: int,
                 n_hidden1_disc: int,
                 n_hidden2_disc: int,
                 dropout: float = 0.0):
        super().__init__()
        self.vgae = VGAE()
        self.discriminator = Discriminator()
        
        self.is_trained_ = False
        self.trainer = None

    def train(self,
              n_epochs: int = 400,
              lr: float = 1e-3,
              eps: float = 0.01,
              **kwargs):
        """
        Train the model.

        Parameters
        ----------
        n_epochs:
            Number of epochs.
        lr:
            Learning rate.
        eps:
            torch.optim.Adam eps parameter.
        kwargs:
            Keyword arguments for the Deeplinc trainer.
        """
        self.trainer = None
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True
        return 1
    def predict():
        return 1