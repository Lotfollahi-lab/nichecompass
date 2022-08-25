import torch


class VGAEModuleMixin:
    """Universal VGAE module functionalities."""
    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        """
        Reparameterization trick for latent space normal distribution.
        """
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
        Map input features x and edge index into the latent space z and return 
        z.
           
        Parameters
        ----------
        x:
            Feature matrix to be mapped into latent space.
        edge_index:
            Corresponding edge index of the graph.

        Returns
        -------
        z:
            Latent space encoding.
        """
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return z

