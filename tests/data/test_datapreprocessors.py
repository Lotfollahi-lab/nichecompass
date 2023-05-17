import numpy as np
import squidpy as sq

from nichecompass.data import load_spatial_adata_from_csv
from nichecompass.data import prepare_data


dataset = "squidpy_seqfish"

print(f"Using dataset {dataset}.")
if dataset == "deeplinc_seqfish":
    adata = load_spatial_adata_from_csv("datasets/seqFISH/counts.csv",
                                        "datasets/seqFISH/adj.csv")
    cell_type_key = None
elif dataset == "squidpy_seqfish":
    adata = sq.datasets.seqfish()
    sq.gr.spatial_neighbors(adata, radius = 0.04, coord_type="generic")
elif dataset == "squidpy_slideseqv2":
    adata = sq.datasets.slideseqv2()
    sq.gr.spatial_neighbors(adata, radius = 30.0, coord_type="generic")

print(f"Number of nodes: {adata.X.shape[0]}")
print(f"Number of node features: {adata.X.shape[1]}")
avg_edges_per_node = round(
    adata.obsp['spatial_connectivities'].toarray().sum(axis=0).mean(),2)
print(f"Average number of edges per node: {avg_edges_per_node}")
n_edges = int(np.triu(adata.obsp['spatial_connectivities'].toarray()).sum())
print(f"Number of edges: {n_edges}", sep="")

data_dict = prepare_data(adata=adata,
                         adj_key="spatial_connectivities",
                         edge_val_ratio=0.1,
                         edge_test_ratio=0.0,
                         node_val_ratio=0.1,
                         node_test_ratio=0.0)

print(f"Edge train data: {data_dict['edge_train_data']}")
print(f"Edge val data: {data_dict['edge_val_data']}")
print(f"Edge test data: {data_dict['edge_test_data']}")
print(f"Edge test data sum: {data_dict['edge_test_data'].edge_label.sum()}")
print(f"Node masked data: {data_dict['node_masked_data']}")
print(f"Node test mask sum: {data_dict['node_masked_data'].test_mask.sum()}")
