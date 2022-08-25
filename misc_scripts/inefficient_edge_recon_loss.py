"""
# this works even in the case of double edges from neg sampling
for i, edge in enumerate(zip(edge_label_index[0], edge_label_index[1])):
    if i == 0:
        edge_recon_logits = torch.unsqueeze(
            adj_recon_logits[edge[0], edge[1]], dim=-1)
    else:
        edge_recon_logits = torch.cat((
            edge_recon_logits,
            torch.unsqueeze(adj_recon_logits[edge[0], edge[1]], dim=-1)))
# Compute cross entropy loss
edge_recon_loss = F.binary_cross_entropy_with_logits(edge_recon_logits,
                                                     edge_labels,
                                                     pos_weight=pos_weight)
"""

# Order edge labels based on sorted edge_label_index to align order with
# masked retrieval from adjacency matrix
sort_index = edge_label_index[0].sort(dim=-1).indices
edge_labels_sorted = edge_labels[sort_index]
