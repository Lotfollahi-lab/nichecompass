from typing import Optional

import numpy as np
import scanpy as sc
import torch

from autotalker.data import SpatialAnnDataset


class VGAEModelMixin:
    """
    
    """
    def get_latent_representation(self, 
                                  x: Optional[torch.Tensor]=None,
                                  edge_index: Optional[torch.Tensor]=None):
        self._check_if_trained(warn=False)
        device = next(self.model.parameters()).device

        if x is not None and edge_index is not None:
            x = torch.tensor(x, device=device)
            edge_index = torch.tensor(edge_index, device=device)
            z = self.model.get_latent_representation(x, edge_index)
        else:
            dataset = SpatialAnnDataset(self.adata, self.adj_key_)
            x = torch.tensor(dataset.x, device=device)
            edge_index = torch.tensor(dataset.edge_index, device=device)

        z = np.array(self.model.get_latent_representation(x, edge_index).cpu())

        return z


    def get_labeled_latent_adata(
            self, 
            x: Optional[torch.Tensor]=None,
            edge_index: Optional[torch.Tensor]=None,
            cell_type_label: Optional[torch.Tensor]=None):
        """
        
        """
        z = self.get_latent_representation(x, edge_index)
        labeled_latent_adata = sc.AnnData(z)
        if cell_type_label is not None:
            labeled_latent_adata.obs["cell_type"] = cell_type_label
        elif self.adata.obs[self.cell_type_key_] is not None:
            labeled_latent_adata.obs["cell_type"] = self.adata.obs[self.cell_type_key_].values
        else:
            raise ValueError("No cell type label found.")
        
        return labeled_latent_adata



                                                    