import anndata as ad
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


class SpatialAnnDataDataset(Dataset):
    """
    Spatial AnnData dataset class.

    Parameters
    ----------
    adata
        Spatially annotated AnnData object. Adjaceny matrix needs to be stored
        in adata.obsp["spatial_connectivities"].
    """

    def __init__(self, adata: ad.AnnData):
        super(SpatialAnnDataDataset, self).__init__()
        if sparse.issparse(adata.X):
            self.X = torch.tensor(adata.X.A)
        else:
            self.X = torch.tensor(adata.X)

        if sparse.issparse(adata.obsp["spatial_connectivities"]):
            # output from squidpy.gr.spatial_neighbors
            self.A = torch.tensor(adata.obsp["spatial_connectivities"].A)
        else:
            self.A = torch.tensor(adata.obsp["spatial_connectivities"])

        if not (self.A == self.A.T).all():
            raise ImportError(
                "The input adjacency matrix is not a symmetric \
                              matrix."
            )
        if not np.diag(self.A).sum() == 0:
            raise ImportError(
                "The diagonal elements of the input adjacency \
                              matrix are not all 0."
            )

    def __getitem__(self, index):
        output = dict()
        output["X"] = self.X[index, :]
        output["A"] = self.A[index, :]
        return output

    def __len__(self):
        return self.X.size(0)

    def __str__(self):
        return self.__name
