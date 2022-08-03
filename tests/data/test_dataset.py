import unittest

import anndata as ad
import numpy as np
import scipy.sparse as sp

from deeplinc.data import SpatialAnnDataDataset
from deeplinc.data import sparse_A_to_edges
from deeplinc.data import has_overlapping_edges


class TestSpatialAnnDataDataset(unittest.TestCase):
    def test_init(self):
        np.random.seed(1)
        n_nodes = 100
        node_dim = 10
        n_edges = 150
        n_nonedges = int(n_nodes ** 2 - n_nodes - n_edges * 2) / 2
        test_ratio = 0.1
        n_edges_test = int(test_ratio * n_edges)
        n_edges_train = n_edges - n_edges_test
        n_edges_test_neg = int(test_ratio * n_edges)
        # Identity feature matrix
        X = np.eye(n_nodes, node_dim).astype("float32")
        print(f"X:\n {X}", "\n")
        
        # Symmetric adjacency matrix
        A = np.random.rand(n_nodes, n_nodes)
        A = (A + A.T)/2
        np.fill_diagonal(A, 0)
        threshold = np.sort(A, axis = None)[-n_edges*2]
        A = (A >= threshold).astype("int")
        print(A.sum())
        print(f"A:\n {A}", "\n")
        
        adata = ad.AnnData(X)
        adata.obsp["spatial_connectivities"] = sp.csr_matrix(A)
        print(f"adata A sparse:\n {adata.obsp['spatial_connectivities']}", "\n")

        dataset = SpatialAnnDataDataset(
            adata,
            A_key = "spatial_connectivities",
            test_ratio = test_ratio)

        print(f"Number of nodes: {dataset.n_nodes}")
        print(f"Number of edges: {dataset.n_edges}")
        print(f"Number of nonedges: {dataset.n_nonedges}")
        print(f"Number of train edges: {dataset.n_edges_train}")
        print(f"Number of test edges: {dataset.n_edges_test}")
        print(f"Number of negative test edges: {dataset.n_edges_test_neg}")
        self.assertEqual(n_nodes, dataset.n_nodes)
        self.assertEqual(n_edges, dataset.n_edges)
        self.assertEqual(n_nonedges, dataset.n_nonedges)
        self.assertEqual(n_edges_train, dataset.n_edges_train)
        self.assertEqual(n_edges_test, dataset.n_edges_test)
        self.assertEqual(n_edges_test_neg, dataset.n_edges_test_neg)
        self.assertEqual(
            int(dataset.A_train.sum()/2),
            len(dataset.edges_train))
        self.assertEqual(
            int(dataset.A_test.sum()/2),
             len(dataset.edges_test))
        self.assertEqual(
            int(dataset.A_train_diag.sum()),
            int(dataset.A_train.sum() + dataset.n_nodes))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_test_neg, 
            dataset.edges_test))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_test, 
            dataset.edges_train))
        self.assertEqual(
            dataset.A_train_diag.sum(),
            dataset.n_nodes + dataset.n_edges_train * 2)
        for i,j in dataset.edges:
            self.assertEqual(dataset.A.toarray()[i, j], 1)
        for i,j in dataset.edges_train:
            self.assertEqual(dataset.A_train[i, j], 1)
        for i,j in dataset.edges_train:
            self.assertEqual(dataset.A_train_diag[i, j], 1)
        for i in range(dataset.n_nodes):
            self.assertEqual(dataset.A_train_diag[i, i], 1)
        for i,j in dataset.edges_train:
            self.assertNotEqual(dataset.A_train_diag_norm[i, j], 0)
        for i in range(dataset.n_nodes):
            self.assertNotEqual(dataset.A_train_diag_norm[i, i], 0)
        for i, j in zip(range(dataset.n_nodes), range(dataset.n_nodes)):
            if (i, j) not in dataset.edges_train and i != j:
                self.assertEqual(dataset.A_train_diag_norm[i, j], 0)
            self.assertNotEqual(dataset.A_train_diag_norm[i, i], 0)
        for i,j in dataset.edges_test:
            self.assertEqual(dataset.A_test[i, j], 1)   
        for i,j in dataset.edges_test_neg:
            self.assertEqual(dataset.A.toarray()[i, j], 0)

        print(type(dataset.A_train_diag))
        print(type(dataset.A_train_diag_norm))

if __name__ == '__main__':
    unittest.main()