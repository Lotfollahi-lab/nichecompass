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