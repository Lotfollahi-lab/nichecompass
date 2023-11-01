import anndata
import torch
import torch_geometric
from numpy import ndarray
import nichecompass as nc

#class SpatialTorchDataset(torch.utils.data.Dataset):
#    """Implementation of the `torch.utils.data.Dataset` interface for spatial multi-omic datasets. This
#    allows the use of the `torch.utils.data.DataLoader` class to create an iterable of the dataset. The
#    input `ndarray` object is converted to a `torch.Tensor`."""
#    #FIXME do we need to maintain this?
#
#    def __init__(self, data_loader, feature_transform=None, label_transform=None):
#        self.features = data_loader.features()
#        self.labels = data_loader.labels()
#        self.feature_transform = feature_transform
#        self.label_transform = label_transform
#
#    def __len__(self) -> int:
#        return len(self.features)
#
#    def __getitem__(self, idx) -> torch.Tensor:
#        feature = torch.from_numpy(self.features[idx, :])
#        label = torch.from_numpy(self.labels[idx, :])
#        if self.feature_transform:
#            feature = self.feature_transform(feature)
#        if self.label_transform:
#            label = self.label_transform(label)
#        return feature, label


#class SpatialGraphDataset(torch_geometric.data.Dataset):
#    """Implementation of the 'torch_geometric.data.Dataset' interface for spatial multi-omic datasets. This
#    allows the use of the 'torch_geometric.loader.DataLoader' class to create an iterable of the dataset.
#    This creates a `torch_geometric.data.Data` object describing a homogeneous graph for each dataset, and
#    saves this to disk."""
#    #FIXME this is the data loader that needs to be used for multiple datasets in an efficient manner
#
#    def __init__(self, data_array):
#        self.data_array = data_array
#
#    @property
#    def raw_file_names(self):
#        pass
#
#    @property
#    def processed_file_names(self):
#        ...
#
#    def download(self):
#        pass
#
#    def len(self):
#        return len(self.data)
#
#    def process(self):
#        for data in self.data_array:
#            graph = torch_geometric.data.Data(
#                x=torch.from_numpy(data.features()),
#                pos=torch.from_numpy(data.coordinates()),
#                y=torch.from_numpy(data.labels())
#            )
#            torch.save()
#
#    def get(self):
#        graph_data = torch.load("data_cache.pt")
#        ...
#        return graph_data


def spatial_graph_data_factory(spatial_data: nc.dataset.SpatialData) -> torch_geometric.data.Data:
    """Uses a spatial data object to generate an in-memory `torch_geometric.data.Data` object."""
    graph = torch_geometric.data.Data(
        x=torch.from_numpy(spatial_data.features()),
        pos=torch.from_numpy(spatial_data.coordinates()),
        y=torch.from_numpy(spatial_data.labels())
    )
    return graph


class SpatialData:
    """Interface for loading spatial multi-omic datasets. Provides methods for accessing coordinate, feature
    and label data from annotated data matrices. Extracts the required arrays from an annotated
    data matrix (`anndata.AnnData`) object."""

    def __init__(self, adata: anndata.AnnData, label_index, spatial_index="spatial"):
        self.data = adata
        self.spatial_index = spatial_index
        self.label_index = label_index

    def coordinates(self) -> ndarray:
        return self.data.obsm[self.spatial_index]

    def features(self) -> ndarray:
        return self.data.X

    def labels(self) -> ndarray:
        return self.data.obs[self.label_index]
