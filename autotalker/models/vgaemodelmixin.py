"""
This module contains generic VGAE functionalities, added as a Mixin to the 
Autotalker model.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from anndata import AnnData

from autotalker.data import SpatialAnnTorchDataset


class VGAEModelMixin:
    """
    VGAE model mix in class for universal VGAE model functionalities.
    """
    def get_latent_representation(
            self, 
            adata: Optional[AnnData]=None,
            counts_key: str="counts",
            adj_key: str="spatial_connectivities",
            use_only_active_gps=True,
            return_mu_std: bool=False
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get latent representation from a trained VGAE model.

        Parameters
        ----------
        adata:
            AnnData object to get the latent representation for if not the one
            passed to the model.
        counts_key:
            Key under which the raw counts are stored in ´adata.layer´.
        adj_key:
            Key under which the sparse adjacency matrix is stored in 
            ´adata.obsp´.
        use_only_active_gps:
            If `True`, only return the latent representation for active gene 
            programs.            
        return_mu_std:
            If `True`, return mu and std instead of a random sample from the
            latent space.

        Returns
        ----------
        z:
            Latent space encoding.
        mu:
            Expected values of the latent posterior.
        std:
            Standard deviations of the latent posterior.
        """
        self._check_if_trained(warn=False)
        device = next(self.model.parameters()).device

        if adata is not None:
            dataset = SpatialAnnTorchDataset(adata, counts_key, adj_key)
        else:
            dataset = SpatialAnnTorchDataset(self.adata,
                                             self.counts_key_,
                                             self.adj_key_)
        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device) 
        if self.model.log_variational:
            x = torch.log(1 + x)

        if return_mu_std:
            mu, std = self.model.get_latent_representation(
                x=x,
                edge_index=edge_index,
                use_only_active_gps=use_only_active_gps,
                return_mu_std=True)
            mu = mu.cpu()
            std = std.cpu()
            return mu, std
        else:
            z = np.array(self.model.get_latent_representation(
                x=x,
                edge_index=edge_index,
                use_only_active_gps=use_only_active_gps,
                return_mu_std=False).cpu())
            return z

    @torch.no_grad()
    def get_gene_expr_dist_params(self, 
                                  adata: Optional[AnnData]=None,
                                  counts_key: str="counts",
                                  adj_key: str="spatial_connectivities",
                                  use_only_active_gps=True,
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ZINB gene expression reconstruction parameters from a trained VGAE 
        model.

        Parameters
        ----------
        adata:
            AnnData object to get the ZINB gene expression parameters for if not
            the one passed to the model.
        counts_key:
            Key under which the raw counts are stored in ´adata.layer´.    
        adj_key:
            Key under which the sparse adjacency matrix is stored in 
            ´adata.obsp´.
        use_only_active_gps:
            If `True`, only return the gene expression distribution 
            parameters for active gene programs.        

        Returns
        ----------
        nb_means:
            Mean parameter of the ZINB for gene expression reconstruction.
        zi_probs:
            Zero inflation probability parameter of the ZINB for gene expression
            reconstruction.
        """
        self._check_if_trained(warn=False)
        device = next(self.model.parameters()).device

        if adata is not None:
            dataset = SpatialAnnTorchDataset(adata, counts_key, adj_key)
        else:
            dataset = SpatialAnnTorchDataset(self.adata,
                                             self.counts_key_,
                                             self.adj_key_)
        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device) 
        
        if self.model.log_variational_:
            x = torch.log(1 + x)

        mu, _ = self.model.get_latent_representation(
            x=x,
            edge_index=edge_index,
            use_only_active_gps=use_only_active_gps,
            return_mu_std=True)
        log_library_size = torch.log(x.sum(1)).unsqueeze(1)
    
        nb_means, zi_probs = self.model.get_gene_expr_dist_params(
            mu,
            log_library_size)
        return nb_means, zi_probs

        if self.gene_expr_recon_dist == "nb":
            nb_means = self.model.get_gene_expr_dist_params(mu,
                                                            log_library_size)
            return nb_means
        if self.gene_expr_recon_dist == "zinb":
            nb_means, zi_prob_logits = self.gene_expr_decoder(z,
                                                              log_library_size)
            zi_probs = torch.sigmoid(zi_prob_logits)
            return nb_means, zi_probs