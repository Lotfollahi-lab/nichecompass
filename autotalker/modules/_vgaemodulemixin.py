import torch


class VGAEModuleMixin:
    """Mixin class containing universal VGAE module functionalities."""
    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        """Reparameterization trick for latent space normal distribution."""
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(mu)
            return eps.mul(std).add(mu)
        else:
            return mu

    @torch.no_grad()
    def get_latent_representation(self,
                                  x: torch.Tensor,
                                  edge_index: torch.Tensor):
        """
        Encode input features x and edge index into the latent space normal 
        distribution parameters and return z. If the module is not in training
        mode, mu will be returned.
           
        Parameters
        ----------
        x:
            Feature matrix to be encoded into latent space.
        edge_index:
            Edge index of the graph.

        Returns
        -------
        z:
            Latent space encoding.
        """
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return z