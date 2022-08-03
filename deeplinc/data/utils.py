import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch


def train_test_split(adata: ad.AnnData,
                     A_key: str = "spatial_connectivities",
                     test_ratio: float = 0.1):
    """
    Splits the edges defined in the adjacency matrix of adata into training and 
    test edges for model training.

    Parameters
    ----------
    adata:
        Spatially annotated AnnData object. Adjaceny matrix with labels and 0s
        on diagonal needs to be stored in ad.AnnData.obsp[A_key].
    A_key:
        Key in ad.AnnData.obsp where adjacency matrix labels are stored. 
        Defaults to"spatial_connectivities", which is where 
        squidpy.gr.spatial_neighbors() outputs the computed adjacency matrix.
    test_ratio:
        Ratio of edges that will be used for testing.
    Returns
    ----------
    A_train:
        Adjacency matrix with training labels and 0s on diagonal.
    A_test:
        Adjacency matrix with testing labels and 0s on diagonal.
    edges_train:
        Numpy array containing training edges.
    edges_test_pos:
        Numpy array containing positive test edges.
    edges_test_neg:
        Numpy array containing negative test edges.
    """
    ## Extract edges from the adjacency matrix
    A = adata.obsp[A_key]
    # Check that diagonal elements of adjacency matrix are set to 0
    if np.diag(A.todense()).sum() != 0:
        raise AssertionError("The diagonal elements of the adjacency matrix \
        are not 0.")
    n_nodes = A.shape[0]
    # Get upper triangle of adjacency matrix (single entry for edges)
    A_triu = sp.triu(A)
    # single edge entry for adjacent cells
    edges_single = sparse_A_to_edges(A_triu)
    # double edge entry for adjacent cells
    edges_double = sparse_A_to_edges(A)
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
    # Get node combinations without edge
    n_nonedges = n_nodes**2-int(A.sum())-n_nodes
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
    A_train = sp.csr_matrix(
        (np.ones(edges_train.shape[0]), (edges_train[:, 0], edges_train[:, 1])),
        shape=A.shape)
    # Make symmetric
    A_train = A_train + A_train.T
    A_test = sp.csr_matrix(
        (np.ones(edges_test.shape[0]), (edges_test[:, 0], edges_test[:, 1])),
        shape=A.shape)
    # Make symmetric
    A_test = A_test + A_test.T

    return A_train, A_test, edges_train, edges_test, edges_test_neg


##### HELPER FUNCTIONS #####

def sparse_A_to_edges(sparse_A):
    """
    Extract node indices of edges from a sparse adjacency matrix.

    Parameters
    ----------
    A
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
    if not sp.isspmatrix_coo(sparse_A):
        sparse_A = sparse_A.tocoo()
    edge_indeces = np.vstack((sparse_A.row, sparse_A.col)).transpose()
    return edge_indeces
    

def sample_neg_edges(n_nodes, edges_pos, edges_excluded):
    """
    Sample as many negative edges as needed to match the number of positive 
    edges. Negative edges connect nodes that are not connected. Self-connecting 
    edges are excluded.

    Parameters
    ----------
    n:
        Number of nodes to sample negative edges from.
    edges_pos:
        Numpy array containing positive edges.
    edges_excluded:
        Numpy array containing edges that are to be excluded.
    Returns
    ----------  
    edges_neg:
        Numpy array containing negative edges.
        Example:
        array([[0,  3],
               [1,  2]], dtype=int32)
    """
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, n_nodes)
        idx_j = np.random.randint(0, n_nodes)
        if idx_i == idx_j:
            continue
        if has_overlapping_edges([idx_i, idx_j], edges_pos):
            continue
        if has_overlapping_edges([idx_j, idx_i], edges_pos):
            continue
        if has_overlapping_edges([idx_i, idx_j], edges_excluded):
            continue
        if has_overlapping_edges([idx_j, idx_i], edges_excluded):
            continue
        if edges_neg:
            if has_overlapping_edges([idx_i, idx_j], np.array(edges_neg)):
                continue
            if has_overlapping_edges([idx_j, idx_i], np.array(edges_neg)):
                continue
        edges_neg.append([idx_i, idx_j])
    edges_neg = np.array(edges_neg, dtype = np.int32)
    return edges_neg


def has_overlapping_edges(edge_array, comparison_edge_array, prec_decimals = 5):
    """
	Check whether two edge arrays have overlapping edges. This is used for 
    sampling of negative edges that are not in positive edge set.

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


def normalize_A(A_diag):
    """
    Symmetrically normalize adjacency matrix as per Kipf, T. N. & Welling, M. 
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016). Calculate
    D**(-1/2)*A*D**(-1/2) where D is the degree matrix and A is the adjacency
    matrix where diagonal elements are set to 1, i.e. every node is connected
    to itself.

    Parameters
    ----------
    A_diag:
        The adjacency matrix to be symmetrically normalized with 1s on diagonal.
    Returns
    ----------  
    A_norm_diag:
        Symmetrically normalized sparse adjacency matrix with diagonal values.
    """
    rowsums = np.array(A_diag.sum(1))  # calculate sums over rows
     # D**(-1/2)
    degree_mx_inv_sqrt = sp.diags(np.power(rowsums, -0.5).flatten())
    # D**(-1/2)*A*D**(-1/2)
    A_norm_diag = (
        A_diag.dot(degree_mx_inv_sqrt).transpose().dot(degree_mx_inv_sqrt).tocoo())
    return sparse_mx_to_sparse_tensor(A_norm_diag)


def sparse_mx_to_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)