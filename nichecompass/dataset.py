import torch
import torch_geometric
from abc import ABC, abstractmethod


class SpatialTorchDataset(torch.utils.data.Dataset):
    """Implementation of the `torch.utils.data.Dataset` interface for spatial multi-omic datasets. This
    allows the use of the `torch.utils.data.DataLoader` class to create an iterable of the dataset."""

    def __init__(self, data_loader, distance_transform=None, molecular_transform=None, target_transform=None):
        self.labels = data_loader.labels()
        self.distance_data = data_loader.distance_data()
        self.molecular_data = data_loader.molecular_data()
        self.distance_transform = distance_transform
        self.molecular_transform = molecular_transform
        self.target_transform = target_transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class SpatialGraphDataset(torch_geometric.data.Dataset):
    """Implementation of the 'torch_geometric.data.Dataset' interface for spatial multi-omic datasets. This
    allows the use of the 'torch_geometric.loader.DataLoader' class to create an iterable of the dataset."""

    def __init__(self, data_loader, distance_transform=None, molecular_transform=None, target_transform=None):
        self.labels = data_loader.labels()
        self.distance_data = data_loader.distance_data()
        self.molecular_data = data_loader.molecular_data()
        self.distance_transform = distance_transform
        self.molecular_transform = molecular_transform
        self.target_transform = target_transform

    def len(self):
        pass

    def get(self):
        pass


class SpatialDataLoader(ABC):
    """Interface for loading spatial multi-omic datasets."""

    @abstractmethod
    def distance_data(self):
        pass

    @abstractmethod
    def molecular_data(self):
        pass

    @abstractmethod
    def labels(self):
        pass


class AnndataLoader(SpatialDataLoader):
    """A data loader for Anndata. Extracts the required datasets from an anndata object"""

    def __init__(self, anndata, molecular_data_indices, label_indices, spatial_transform):
        self.data = anndata
        self.molecular_data_indices = molecular_data_indices
        self.label_indices = label_indices
        self.spatial_transform = spatial_transform

    def distance_data(self):
        return self.spatial_transform(self.data)

    def molecular_data(self):
        pass

    def labels(self):
        pass


class DelimLoader(SpatialDataLoader):
    """A data loader for delim"""
