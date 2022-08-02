import squidpy as sq
import torch
import scipy.sparse as sp
import numpy as np
adata = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata, n_rings=2, coord_type="grid", n_neighs=200)
adj_mtx = adata.obsp["spatial_connectivities"]
type(adata.X)
sparse_mx = adata.X.tocoo()
sparse_mx.shape
values = sparse_mx.data
indices = np.vstack((sparse_mx.row, sparse_mx.col))
shape = sparse_mx.shape
a= torch.FloatTensor(values)
b= torch.IntTensor(indices)
test = torch.sparse_coo_tensor(b, a, torch.Size(shape))
print(test)