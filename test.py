from deeplinc.data import simulate_spatial_adata

adata = simulate_spatial_adata(n_node_features=50)
print(adata.X)