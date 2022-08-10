import squidpy as sq

from autotalker.autotalker.data._datasets import SpatialAnnDataDataset

adata = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata, n_rings=2, coord_type="grid", n_neighs=10)

dataset = SpatialAnnDataDataset(
    adata,
    A_key = "spatial_connectivities",
    test_ratio = 0.1)

print(f"Number of nodes: {dataset.n_nodes}")
print(f"Number of total edges in A: {dataset.n_edges}")
print(f"Number of train edges: {dataset.n_edges_train}")
print(f"Number of positive test edges: {dataset.n_edges_test}")
print(f"Number of negative test edges: {dataset.n_edges_test_neg}")
print(f"Number of total edges in A train: {dataset.A_train.sum()}")
print(f"Number of total edges in A test: {dataset.A_test.sum()}")