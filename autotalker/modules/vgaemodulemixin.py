import torch


class VGAEModuleMixin:
    """
    VGAE module mix in class containing universal VGAE module 
    functionalities.
    """
    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        """
        Use reparameterization trick for latent space normal distribution.
        
        Parameters
        ----------
        mu:
            Expected values of the latent space distribution.
        logstd:
            Log standard deviations of the latent space distribution.

        Returns
        ----------
        rep:
            Reparameterized latent space values.
        """
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(mu)
            rep = eps.mul(std).add(mu)
            return rep
        else:
            rep = mu
            return rep

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

    @torch.no_grad()
    def get_reconstructed_adj(self,
                              z: torch.Tensor,
                              edge_prob_threshold: float):
        """
        
        """
        adj_recon_probs = torch.sigmoid(self.graph_decoder(z))
        adj_recon = (adj_recon_probs > edge_prob_threshold).astype("int")
        return adj_recon