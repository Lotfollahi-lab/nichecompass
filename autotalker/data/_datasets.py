import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from ._utils import sparse_A_to_edges
from ._utils import has_overlapping_edges
from ._utils import sample_neg_edges
from ._utils import sparse_mx_to_sparse_tensor
from ._utils import normalize_A


class SpatialAnnDataset():
    def __init__(
            self,
            adata: ad.AnnData,
            adj_key: str = "spatial_connectivities"):

        # Store features in dense format
        if sp.issparse(adata.X): 
            self.x = torch.FloatTensor(adata.X.toarray())
        else:
            self.x = torch.FloatTensor(adata.X)

        self.n_node_features = self.x.size(1)

        # Store adjacency matrix in sparse tensor format
        self.adj = sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        if not (self.adj.to_dense() == self.adj.to_dense().T).all():
            raise ImportError("The input adjacency matrix is not symmetric.")

        self.edge_index = self.adj._indices()

            


class SpatialAnnDataPyGDataset(Dataset):
    """
    Dataset handler for autotalker model and trainer.

    Parameters
    ----------
    adata:
        Spatially annotated AnnData object. Adjaceny matrix needs to be stored 
        in ad.AnnData.obsp[A_key].
    A_key:
        Key in ad.AnnData.obsp where adjacency matrix is stored. Defaults to
        "spatial_connectivities", which is where squidpy.gr.spatial_neighbors()
        outputs computed adjacency matrix.
    test_ratio:
        Ratio of edges that will be used for testing.
    """
    def __init__(
            self,
            adata: ad.AnnData,
            A_key: str = "spatial_connectivities",
            test_ratio: float = 0.1):
        super().__init__()

        # Store features in dense format
        if sp.issparse(adata.X): 
            self.X = torch.FloatTensor(adata.X.toarray())
        else:
            self.X = torch.FloatTensor(adata.X)

        self.n_node_features = self.X.size(1)
        
        # Store adjacency matrix in sparse format
        if not sp.isspmatrix_coo(adata.obsp[A_key]):
            self.A = adata.obsp[A_key].tocoo()
        else:
            self.A = adata.obsp[A_key]
        if not (self.A.todense() == self.A.todense().T).all():
            raise ImportError("The input adjacency matrix is not symmetric.")
        #if not np.diag(self.A.todense()).sum() == 0:
        #    raise ImportError("The diagonal elements of the input adjacency \
        #                      matrix are not all 0.")

        self.A_diag = self.A + sp.eye(self.A.shape[0])

        self.n_nodes = int(self.A.shape[0])
        self.n_edges = int(self.A.todense().sum()/2)
        self.n_nonedges = int(
            (self.n_nodes ** 2 - self.n_edges * 2 - self.n_nodes) / 2)
        if test_ratio > 1: # absolute test ratio
            self.test_ratio = test_ratio / self.n_edges 
        else:
            self.test_ratio = test_ratio
        self.n_edges_test = int(np.floor(self.n_edges * self.test_ratio))
        if self.n_nonedges < 2 * self.n_edges_test:
            raise ImportError("The network is too dense. Please decrease the \
            test ratio or delete some edges in the network.")
        
        extracted_edges = self._extract_edges()
        self.edges = extracted_edges[0] 
        self.edges_train = extracted_edges[1] 
        self.edges_train_neg = extracted_edges[2] 
        self.edges_test = extracted_edges[3] 
        self.edges_test_neg = extracted_edges[4]

        self.n_edges_train = len(self.edges_train)
        self.n_edges_train_neg = len(self.edges_train_neg)
        self.n_edges_test_neg = len(self.edges_test_neg)

        preprocessed_As = self._preprocess_A()
        self.A_train = preprocessed_As[0]
        self.A_train_diag = preprocessed_As[1]
        self.A_train_diag_norm = preprocessed_As[2]
        self.A_test = preprocessed_As[3]

    def __getitem__(self, index):
        output = dict()
        output["X"] = self.X[index, :]
        return output

    def __len__(self):
        return self.X.size(0)


    def __str__(self):
        return self.__name


    def _extract_edges(self):
        """
        Extract node index pairs of edges from self.A, which is a sparse 
        adjacency matrix in coo format with 0s on the diagonal.
        
        Returns
        ----------
        edges_train:
            Numpy array containing training edges.
        edges_test_pos:
            Numpy array containing positive test edges.
        edges_test_neg:
            Numpy array containing negative test edges.
        """
        # Get upper triangle of adjacency matrix (single entry for edges)
        A_triu = sp.triu(self.A)
        edges = sparse_A_to_edges(A_triu) # single edge adjacent nodes
        edges_double = sparse_A_to_edges(self.A) # double edge adjacent nodes
        idx_edges_all = np.array(range(self.n_edges))
        np.random.shuffle(idx_edges_all)
        idx_edges_test = idx_edges_all[:self.n_edges_test]
        idx_edges_train = idx_edges_all[self.n_edges_test:]

        edges_test = edges[idx_edges_test]
        edges_train = edges[idx_edges_train]
        edges_test_neg = sample_neg_edges(
            n_nodes = self.n_nodes,
            edges_pos = edges_test,
            edges_excluded = edges_train)
        edges_train_neg = sample_neg_edges(
            n_nodes = self.n_nodes,
            edges_pos = edges_train,
            edges_excluded = np.concatenate(
                (edges_test, edges_test_neg), axis=0))

        # Sort edge arrays
        edges_test = edges_test[np.lexsort((edges_test[:,1], edges_test[:,0]))]
        edges_train = edges_train[np.lexsort(
            (edges_train[:,1], edges_train[:,0]))]
        edges_test_neg = edges_test_neg[np.lexsort(
            (edges_test_neg[:,1], edges_test_neg[:,0]))]
        edges_train_neg = edges_train_neg[np.lexsort(
            (edges_train_neg[:,1], edges_train_neg[:,0]))]

        assert ~has_overlapping_edges(edges_test_neg, edges_double)
        assert ~has_overlapping_edges(edges_train_neg, edges_double)

        return edges, edges_train, edges_train_neg, edges_test, edges_test_neg


    def _preprocess_A(self):
        """
        Preprocess the adjacency matrix for model training.

        Returns
        ----------
        A_train:
            Adjacency matrix with training labels and 0s on diagonal.
        A_train_diag:
            Adjacency matrix with training labels and 1s on diagonal.
        A_train_diag_norm:
            Symmetrically normalized training adjacency matrix with values on 
            diagonal.
        """
        # Construct (symmetric) train and test adjacency matrices
        tmp = np.ones(self.edges_train.shape[0])
        A_train = sp.csr_matrix(
            (tmp, (self.edges_train[:, 0], self.edges_train[:, 1])),
            shape = self.A.shape)
        # Make symmetric
        A_train = A_train + A_train.T

        tmp = np.ones(self.edges_test.shape[0])
        A_test = sp.csr_matrix(
            (tmp, (self.edges_test[:, 0], self.edges_test[:, 1])),
            shape = self.A.shape)
        # Make symmetric
        A_test = A_test + A_test.T

        A_train_diag = A_train + sp.eye(A_train.shape[0])
        # Store as sparse tensor
        A_train_diag_norm = normalize_A(A_train_diag)
        
        # Store as sparse tensor
        A_train = sparse_mx_to_sparse_tensor(A_train)
        A_train_diag = sparse_mx_to_sparse_tensor(A_train_diag)
        A_train_diag_norm = sparse_mx_to_sparse_tensor(A_train_diag_norm)
        A_test = sparse_mx_to_sparse_tensor(A_test)

        return A_train, A_train_diag, A_train_diag_norm, A_test