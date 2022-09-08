from typing import Optional

import numpy as np
from anndata import AnnData

from autotalker.data import SpatialAnnTorchDataset


class VGAEModelMixin:
    """
    VGAE model mix in class for universal VGAE model functionalities.
    """
    def get_latent_representation(self, 
                                  adata: Optional[AnnData]=None,
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
            dataset = SpatialAnnTorchDataset(adata, adj_key)
        else:
            dataset = SpatialAnnTorchDataset(self.adata, self.adj_key_)

        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device) 

        z = np.array(self.model.get_latent_representation(x, edge_index).cpu())
        return z



                                                    