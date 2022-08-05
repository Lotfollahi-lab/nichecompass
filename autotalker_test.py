from autotalker.data import simulate_spatial_adata

adata = simulate_spatial_adata(
    n_node_features=50,
    adj_nodes_feature_multiplier=1000,
    debug=True)