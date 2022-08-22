from typing import Optional

import anndata as ad
import numpy as np

from autotalker.data import SpatialAnnDataset


class VGAEModelMixin:
    """
    Universal VGAE model functionalities.
    """
    def get_latent_representation(self, 
                                  adata: Optional[ad.AnnData]=None,
                                  adj_key: str="spatial_connectivities"):
        """
        Get latent representation from trained VGAE model.
        """
        self._check_if_trained(warn=False)
        device = next(self.model.parameters()).device

        if adata is not None:
            dataset = SpatialAnnDataset(adata, adj_key)
        else:
            dataset = SpatialAnnDataset(self.adata, self.adj_key_)

        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device) 

        z = np.array(self.model.get_latent_representation(x, edge_index).cpu())

        return z



                                                    