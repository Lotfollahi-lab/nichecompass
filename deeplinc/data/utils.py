import anndata as ad
import numpy as np
import scipy.sparse as sp


def train_test_split(adata: ad.AnnData,
                     adj_mx_key: str = "spatial_connectivities",
                     test_ratio: float = 0.1):
    """
    Splits the edges defined in the adjacency matrix of adata into training and 
    test edges for model training.

    Parameters
    ----------
    adata
        Spatially annotated AnnData object. Adjaceny matrix needs to be stored 
        in ad.AnnData.obsp[adj_mx_key].
    adj_mx_key
        Key in ad.AnnData.obsp where adjacency matrix is stored. Defaults to
        "spatial_connectivities", which is where squidpy.gr.spatial_neighbors()
        outputs the computed adjacency matrix.
    test_ratio
        Ratio of edges that will be used for testing.
    Returns
    ----------
    adj_mx_train
    adj_mx_test
    edges_train
    edges_test
    edges_test_neg
    """
    ## Extract edges from the adjacency matrix
    adj_mx = adata.obsp[adj_mx_key]
    # Check that diagonal elements of adjacency matrix are set to 0
    if np.diag(adj_mx.todense()).sum() != 0:
        raise AssertionError("The diagonal elements of the adjacency matrix \
        are not 0.")
    n_nodes = adj_mx.shape[0]
    adj_mx_triu = sp.triu(adj_mx)  # upper triangle of adjacency matrix
    # single edge for adjacent cells
    edges_single = sparse_adj_mx_to_edges(adj_mx_triu)
    # double edge for adjacent cells
    edges_double = sparse_adj_mx_to_edges(adj_mx)
    n_edges = edges_single.shape[0]

    if test_ratio > 1: # absolute test ratio
        test_ratio = test_ratio / n_edges 

    ## Split into train and test edges
    n_edges_test = int(np.floor(n_edges * test_ratio))
    idx_edges_all = np.array(range(n_edges))
    np.random.shuffle(idx_edges_all)
    idx_edges_test = idx_edges_all[:n_edges_test]
    edges_test = edges_single[idx_edges_test]
    edges_train = np.delete(edges_single, idx_edges_test, axis=0)

    ## Sample negative test edges
    # node combinations without edge
    n_nonedges = n_nodes**2-int(adj_mx.sum())-n_nodes
    if (n_nonedges)/2 < 2*n_edges_test:
        raise AssertionError("The network is too dense. Please decrease the \
        test ratio or delete some edges in the network.")
    else:
        edges_test_neg = sample_neg_test_edges(n_nodes,
                                               edges_test,
                                               edges_double)

    assert ~has_overlapping_edges(edges_test_neg, edges_double)
    assert ~has_overlapping_edges(edges_test, edges_train)

    ## Construct (symmetric) train and test adjacency matrix with sampled edges
    adj_mx_train = sp.csr_matrix(
        (np.ones(edges_train.shape[0]), (edges_train[:, 0], edges_train[:, 1])),
        shape=adj_mx.shape,
    )
    adj_mx_train = adj_mx_train + adj_mx_train.T # make symmetric
    adj_mx_test = sp.csr_matrix(
        (np.ones(edges_test.shape[0]), (edges_test[:, 0], edges_test[:, 1])),
        shape=adj_mx.shape,
    )
    adj_mx_test = adj_mx_test + adj_mx_test.T # make symmetric

    return adj_mx_train, adj_mx_test, edges_train, edges_test, edges_test_neg


##### HELPER FUNCTIONS #####

def sparse_adj_mx_to_edges(sparse_adj_mx):
    """
    Extract node indices of edges from a sparse adjacency matrix.

    Parameters
    ----------
    adj_mx
        Sparse adjacency matrix from which edges are to be extracted.
    Returns
    ----------
    edge_indeces
        Numpy array containing node indices of edges.
        Example:
        array([[0, 1],
               [0, 2],
               [1, 0],
               [2, 0],
               [3, 4],
               [4, 3]], dtype=int32)
    """
    if not sp.isspmatrix_coo(sparse_adj_mx):
        sparse_adj_mx = sparse_adj_mx.tocoo()
    edge_indeces = np.vstack((sparse_adj_mx.row, sparse_adj_mx.col)).transpose()
    return edge_indeces
    

def sample_neg_test_edges(n_nodes, edges_test, edges_double):
    """
    Sample as many negative test edges as needed to match the number of positive
    test edges. Negative test edges connect nodes that are not connected.

    Parameters
    ----------
    n
        Number of nodes to sample negative test edges from.
    edges_test
        Numpy array containing positive test edges.
    edges_double
        Numpy array containing all existing edges with double directions
    Returns
    ----------  
    edges_test_neg
        Numpy array containing negative test edges.
        Example:
        array([[0,  3],
               [3,  0]], dtype=int32)
    """
    edges_test_neg = []
    while len(edges_test_neg) < len(edges_test):
        idx_i = np.random.randint(0, n_nodes)
        idx_j = np.random.randint(0, n_nodes)
        if idx_i == idx_j:
            continue
        if has_overlapping_edges([idx_i, idx_j], edges_double):
            continue
        if has_overlapping_edges([idx_j, idx_i], edges_double): # redundant
            continue
        if edges_test_neg: # if list contains values
            if has_overlapping_edges([idx_j, idx_i], np.array(edges_test_neg)):
                continue
            if has_overlapping_edges([idx_i, idx_j], np.array(edges_test_neg)):
                continue
        edges_test_neg.append([idx_i, idx_j])
    edges_test_neg = np.array(edges_test_neg, dtype=np.int32)
    return edges_test_neg


def has_overlapping_edges(edge_array, comparison_edge_array, prec_decimals = 5):
    """
	Check whether two edge arrays have overlapping edges. This is used for 
    sampling of negative test edges that are not in positive test set.

    Parameters
    ----------
    edge_array
        Numpy array of edges to be tested for overlap.
    comparison_edge_array
        Numpy array of comparison edges to be tested for overlap.
    prec_decimals
        Decimals for overlap precision.
    Returns
    ----------
    overlap
        Boolean that indicates whether the two edge arrays have an overlap. 
	"""
    edge_overlaps = np.all(np.round(edge_array - comparison_edge_array[:, None],
                                    prec_decimals) == 0,
                           axis=-1)
    if True in np.any(edge_overlaps, axis=-1).tolist():
        overlap = True
    elif True not in np.any(edge_overlaps, axis=-1).tolist():
        overlap = False
    return overlap


def normalize_adj_mx(adj_mx):
    """
    Symmetrically normalize adjacency matrix as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016). Calculate
    D**(-1/2)*A*D**(-1/2) where D is the degree matrix and A is the adjacency
    matrix where diagonal elements are set to 1, i.e. every node is connected
    to itself.

    Parameters
    ----------
    adj_mx
        The adjacency matrix to be symmetrically normalized.
    Returns
    ----------  
    adj_mx_nom
        Symmetrically normalized sparse adjacency matrix.
    """
    adj_mx = sp.coo_matrix(adj_mx)  # convert to sparse matrix COOrdinate format
    adj_mx_ = adj_mx + sp.eye(adj_mx.shape[0])  # add 1s on diagonal
    rowsums = np.array(adj_mx_.sum(1))  # calculate sums over rows
    degree_mx_inv_sqrt = sp.diags(np.power(rowsums, -0.5).flatten())  # D**(-1/2)
    adj_mx_norm = (
        adj_mx_.dot(degree_mx_inv_sqrt).transpose().dot(degree_mx_inv_sqrt).tocoo()
    )  # D**(-1/2)*A*D**(-1/2)
    return adj_mx_norm


def test(adata: ad.AnnData,
                adj_mx_key: str = "spatial_connectivities",
                test_ratio: float = 0.1):

    split_adj_mx_and_edges = train_test_split(adata=adata,
                                              adj_mx_key=adj_mx_key,
                                              test_ratio=test_ratio)
    adj_mx_train, adj_mx_test = split_adj_mx_and_edges[:2]
    edges_train, edges_test, edges_test_neg = split_adj_mx_and_edges[2:]

    n_nodes = adj_mx_train.shape[0]
    
    adj_mx_labels = adj_mx_train + sp.eye(adj_mx_train.shape[0])


    return


def make_dataset(
    adata,
    train_ratio,
):
    """
    Splits adata into train and validation data.

    Parameters
    ----------
    """
    return 1
