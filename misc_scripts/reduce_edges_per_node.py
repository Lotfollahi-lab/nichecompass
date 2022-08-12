def reduce_edges_per_node(A_rec_logits,
                          optimal_threshold,
                          edges_per_node,
                          reduction):
    """
    Reduce the edges of the reconstruced edge probability adjacency matrix to
    best_edge_target per node.

    Parameters
    ----------
    A_rec_logits:
        Reconstructed adjacency matrix with logits.
    optimal_threshold:
        Optimal classification threshold as calculated with 
        get_optimal_cls_threshold_and_accuracy().
    edges_per_node:
        Target for edges per node.
    reduction:
        "soft": Keep edges that are among the top <edges_per_node> edges for one
        of the two edge nodes.
        "hard": Keep edges that are among the top <edges_per_node> edgesfor both
        of the two edge nodes.
    Returns
    ----------
    A_rec_new
        The new reconstructed adjacency matrix with reduced edge predictions.
    """
    # Calculate adjacency matrix with edge probabilities
    adj_rec_probs = torch.sigmoid(A_rec_logits)
    A_rec = copy.deepcopy(adj_rec_probs)
    A_rec = (A_rec>optimal_threshold).int()
    A_rec_tmp = copy.deepcopy(adj_rec_probs)
    for node in range(0, A_rec_tmp.shape[0]):
        tmp = A_rec_tmp[node,:]
        A_rec_tmp[node,:] = (A_rec_tmp[node,:] >= np.sort(tmp)[-edges_per_node]).int()
    A_rec_new = A_rec + A_rec_tmp
    # Mark edges that have been recreated and are among the top 2 node edges for
    # one of the nodes
    A_rec_new = (A_rec_new == 2).int()
    # Make adjacency matrix symmetric
    A_rec_new = A_rec_new + A_rec_new.T 
    # keep edges that are among the top 2 node edges for one of the nodes
    if reduction == "soft": # union
        A_rec_new = (A_rec_new != 0).int()
    # keep edges that are among the top 2 node edges for both of the nodes
    elif reduction == "hard": # intersection
        A_rec_new = (A_rec_new == 2).int()
    return A_rec_new