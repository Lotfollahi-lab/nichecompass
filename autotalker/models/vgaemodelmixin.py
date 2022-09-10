from typing import Optional
from xml.etree.ElementPath import xpath_tokenizer

import numpy as np
import torch
from anndata import AnnData

from autotalker.data import SpatialAnnTorchDataset


class VGAEModelMixin:
    """
    VGAE model mix in class for universal VGAE model functionalities.
    """
    def get_latent_representation(self, 
                                  adata: Optional[AnnData]=None,
                                  counts_layer_key: str="counts",
                                  adj_key: str="spatial_connectivities"):
        """
        Get latent representation from a trained VGAE model.

        Parameters
        ----------
        adata:
            AnnData object to get the latent representation for if not the one
            passed to the model.
        adj_key:
            Key under which the sparse adjacency matrix is stored in 
            ´adata.obsp´.

        Returns
        ----------
        z:
            Numpy array containing latent dimensions.
        """
        self._check_if_trained(warn=False)
        device = next(self.model.parameters()).device

        if adata is not None:
            dataset = SpatialAnnTorchDataset(adata, counts_layer_key, adj_key)
        else:
            dataset = SpatialAnnTorchDataset(self.adata,
                                             self.counts_layer_key_,
                                             self.adj_key_)

        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device) 
        
        if self.model.log_variational:
            x = torch.log(1 + x) # for numerical stability during model training

        z = np.array(self.model.get_latent_representation(x, edge_index).cpu())
        return z