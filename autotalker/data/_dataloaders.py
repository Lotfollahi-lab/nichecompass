import torch_geometric


def initialize_link_level_dataloader(data: torch_geometric.data.Data,
                                     batch_size: int,
                                     n_direct_neighbors: int=-1,
                                     n_neighbor_iters: int=3,
                                     directed=False,
                                     neg_sampling_ratio=1.0):
    """
    Parameters
    ----------
    data:
        PyG Data object to initialize a dataloader for.
    batch_size:
        Ratio of edges to be included in validation split.
    n_direct_neighbors:
        Number of direct neighbors of the edge nodes to be included in batch.
        Defaults to ´-1´, which means to include all direct neighbors.
    n_neighbor_iters:
        Number of neighbor iterations/levels to be included in batch. E.g. ´2´
        means to not only include direct neighbors of edge nodes but also the
        neighbors of the direct neighbors.
    directed:
        If `False` both symmetric edge index pairs are included in the same 
        batch (1 edge has 2 symmetric edge index pairs).
    neg_sampling_ratio:
        Negative sampling ratio of edges. This is currently implemented in an
        approximate way, i.e. negative edges may contain false negatives.

    Returns
    ----------
    dataloader:
        PyG LinkNeighborLoader object.
    """

    dataloader = torch_geometric.loader.LinkNeighborLoader(
        data,
        num_neighbors=[n_direct_neighbors] * n_neighbor_iters,
        batch_size=batch_size,
        edge_label_index=data.edge_label_index,
        directed=directed, 
        neg_sampling_ratio=neg_sampling_ratio)

    return dataloader