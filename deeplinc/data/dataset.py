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
        in ad.AnnData.obsp[adj_mx_key].
    adj_mx_key
        Key in ad.AnnData.obsp where adjacency matrix is stored. Defaults to
        "spatial_connectivities", which is where squidpy.gr.spatial_neighbors()
        outputs computed adjacency matrix.
    """

    def __init__(self,
                 adata: ad.AnnData,
                 adj_mx_key: str = "spatial_connectivities"):
        super(SpatialAnnDataDataset, self).__init__()
        if sparse.issparse(adata.X):  # keep sparsity?
            self.X = torch.tensor(adata.X.A)
        else:
            self.X = torch.tensor(adata.X)

        if sparse.issparse(adata.obsp[adj_mx_key]):
            # output from squidpy.gr.spatial_neighbors
            self.A = torch.tensor(adata.obsp[adj_mx_key].A)
        else:
            self.A = torch.tensor(adata.obsp[adj_mx_key])

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
