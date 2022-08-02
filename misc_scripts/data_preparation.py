import squidpy as sq
import torch
import scipy.sparse as sp
import numpy as np

from deeplinc.data.utils import train_test_split
from deeplinc.data.utils import normalize_A

adata = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata, n_rings=2, coord_type="grid", n_neighs=10)
A = adata.obsp["spatial_connectivities"]
n_nodes = A.toarray().shape[0]
A_and_edges_train_test_split = train_test_split(adata, "spatial_connectivities")
A_train, A_test = A_and_edges_train_test_split[:2]
edges_train, edges_test_pos, edges_test_neg = A_and_edges_train_test_split[2:]

A_train_norm = normalize_A(A_train)
A_label = A_train + sp.eye(A_train.shape[0])
A_label = torch.FloatTensor(A_label.toarray())

print(f"Number of nodes: {n_nodes}")
print(f"Number of total edges in A: {int(A.sum())}")
print(f"Number of train edges: {len(edges_train)}")
print(f"Number of positive test edges: {len(edges_test_pos)}")
print(f"Number of negative test edges: {len(edges_test_neg)}")
print(f"Number of total edges in A train: {int(A_train.sum())}")
print(f"Number of total edges in A test: {int(A_test.sum())}")
#print(A)
#print("")
#print(A_train)
#print("")
#print(edges_train)
print(A_train_norm)
print(f"Number of total edges in A_label: {int(A_label.sum())}")


