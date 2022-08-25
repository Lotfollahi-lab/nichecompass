if self_loops:
    # Add remaining self-loops to ´edge_label_index´ and add corresponding 
    # positive edge labels to ´edge_labels´. Remaining self-loops is used as
    # self loops might occur in negative sampling.
    n_edges_no_self_loops = edge_label_index.shape[1]
    edge_label_index = add_remaining_self_loops(edge_label_index)[0]
    n_self_loops = edge_label_index.shape[1] - n_edges_no_self_loops
    edge_labels = torch.cat((edge_labels, torch.ones(n_self_loops)))
"""self_loops:
    If ´True´ include self-loops as positive edges in edge reconstruction 
    loss.
"""

    """
    adj_labels = to_dense_adj(edge_label_index, max_num_nodes=n_nodes)
    
    # Create mask to retrieve values at edge_label_index from ´adj_recon_logits´
    mask = torch.squeeze(adj_labels > 0)

    # Retrieve logits for edges from ´adj_recon_logits´
    edge_recon_logits = torch.masked_select(adj_recon_logits, mask)

    if mask.sum().item() == 127.:
        print((adj_labels == 2).nonzero(as_tuple=True))
        print(adj_labels[0,193,196:198])
        print(edge_label_index.T)
    """