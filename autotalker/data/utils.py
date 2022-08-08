import sys

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
import scipy.stats as st
import squidpy as sq
import torch


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
        A_diag.dot(degree_mx_inv_sqrt).transpose().dot(degree_mx_inv_sqrt))
    return A_norm_diag


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


def load_spatial_adata_from_csv(
         X_file_path,
         A_file_path,
         A_key = "spatial_connectivities"):
    """
    
    """
    adata = ad.read_csv(X_file_path)
    A_df = pd.read_csv(A_file_path, sep=",", header = 0)
    A = A_df.values
    adata.obsp[A_key] = sp.csr_matrix(A).tocoo()

    return adata


def simulate_spatial_adata(
        n_nodes = 100,
        n_node_features = 0,
        adj_nodes_feature_multiplier = 3,
        n_edges = 150,
        random_seed = 1,
        debug = False):
    """
    Simulate (correlated) feature and adjacency matrices and return as adata.

    Parameters
    ----------
    n_nodes:
        Number of simulated nodes.
    n_node_features:
        Number of simulated node features. If == 0, identity matrix will be used
        as feature matrix as per Kipf, T. N. & Welling, M. Variational Graph 
        Auto-Encoders. arXiv [stat.ML] (2016).
    adj_nodes_feature_multiplier:
        Multiplier to increase feature correlation for adjacent nodes and their
        joint neighbors.
    n_edges:
        Number of simulated edges.
    random_seed:
        Random seed used for generation of random numbers.
    """
    np.random.seed(random_seed)
    
    # Create symmetric adjacency matrix
    A = np.random.rand(n_nodes, n_nodes)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    threshold = np.sort(A, axis = None)[-(n_edges * 2)]
    A = (A >= threshold).astype("int")

    if debug:
        print("")
        print(f"A:\n {A}", "\n")

    if n_node_features == 0: # identity matrix as feature matrix
        X = np.eye(n_nodes, n_nodes).astype("float32")
    else:
        X = np.random.rand(n_nodes, n_node_features)
        # Increase feature correlation between adjacent nodes and their joint
        # neighbors
        for i in range(n_nodes):
            for j in range(n_nodes):
                tmp = adj_nodes_feature_multiplier * np.random.rand(
                    1, n_node_features)
                if A[i, j] == 1:
                    X[i, :] = X[i, :] + tmp
                    X[j, :] = X[j, :] + tmp
                    joint_neighb_idces = np.where(A[i, :] * A[j, :] == 1)[0]
                    for joint_neighb_idx in joint_neighb_idces:
                        X[joint_neighb_idx, :] = X[joint_neighb_idx, :] + tmp
        
        # Normalize feature matrix node-wise
        rowsums = X.sum(axis = 1)
        X = X / rowsums[:, np.newaxis]

        # Calculate correlation of adjacent nodes vs non-adjacent nodes
        corr_adj = 0
        corr_non_adj = 0 
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue 
                if A[i, j] == 1:
                    corr_adj += st.stats.pearsonr(X[i, :], X[j, :])[0]
                if A[i, j] != 1:
                    corr_non_adj += st.stats.pearsonr(X[i, :], X[j, :])[0]
        mean_corr_adj = corr_adj / (A.sum() / 2)
        mean_corr_non_adj = corr_non_adj / (n_nodes ** 2 - A.sum() - n_nodes)

    if debug:
        print(f"X:\n {X}", "\n")
        print(f"Average correlation adjacent nodes: {mean_corr_adj}")
        print(f"Average correlation non-adjacent nodes: {mean_corr_non_adj}")

    adata = ad.AnnData(X.astype("float32"))
    adata.obsp["spatial_connectivities"] = sp.csr_matrix(A)

    return adata


def load_benchmark_spatial_adata(dataset = "cora"):
    # Load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("datasets/benchmark/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("datasets/benchmark/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    adata = ad.AnnData(features.toarray().astype("float32"))
    adata.obsp["spatial_connectivities"] = sp.csr_matrix(adj)

    return adata
    

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index