import torch

from autotalker.data import load_spatial_adata_from_csv
from autotalker.data import SpatialAnnDataset
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_dense_adj

def prepare_data(
    adata,
    val_frac: float = 0.1,
    test_frac: float = 0.1):

    dataset = SpatialAnnDataset(adata)
    data = Data(x = dataset.x, edge_index = dataset.edge_index)

    transform = RandomLinkSplit(
        num_val = val_frac,
        num_test = test_frac,
        is_undirected = True,
        split_labels = True)

    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data

adata = load_spatial_adata_from_csv(
    "datasets/MERFISH/counts.csv",
    "datasets/MERFISH/adj.csv")

train, val, test = prepare_data(adata)

train.edge_index = add_self_loops(train.edge_index)[0]
print(to_dense_adj(train.edge_index))

