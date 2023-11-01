import torch




class VariationalAutoencoderModel(torch.nn.Module):
    #FIXME can't we just use the model from pytorch_geometric ?
    """A generic variational autoencoder module, which can be configured with
    `torch.nn.Module` to describe the encoder, latent and decoder."""

    def __int__(self, encoder: torch.nn.Module, latent: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder

    def forward(self):
        pass



# WE SHOULD USE THE IN-BUILT VAE IN PYTORCH_GEOMETRIC!










def model_vae_factory(Config):
    """This factory uses the run configuration to build a model."""
    ...


### because it is just a module, we can print the entire architecture with print()





