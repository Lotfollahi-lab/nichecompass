import squidpy as sq
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader

from autotalker.data import SpatialAnnDataset

adata = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata, radius = 0.04, coord_type="generic")

print("Average number of neighbours: "
      f"{adata.obsp['spatial_connectivities'].toarray().sum(axis=0).mean()}")
print(f"Number of nodes: {adata.X.shape[0]}")
print(f"Number of node features: {adata.X.shape[1]}")
print("Number of edges: "
      f"{int(np.triu(adata.obsp['spatial_connectivities'].toarray()).sum())}",
      sep="")

dataset = SpatialAnnDataset(adata)
data = Data(x=dataset.x, edge_index=dataset.edge_index)

loader = LinkNeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[1] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=8,
    directed=False,
    neg_sampling_ratio=1)

sampled_data = next(iter(loader))
print(sampled_data)
print(sampled_data.edge_index)
print(sampled_data.edge_label_index)
print(sampled_data.edge_label)

sampled_data = next(iter(loader))
print(sampled_data)
print(sampled_data.edge_index)
print(sampled_data.edge_label_index)
print(sampled_data.edge_label)
