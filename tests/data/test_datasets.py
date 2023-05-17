import unittest

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from nichecompass.data import SpatialAnnDataDataset
from nichecompass.data import sparse_A_to_edges
from nichecompass.data import has_overlapping_edges
from nichecompass.data import simulate_spatial_adata


class TestSpatialAnnDataDataset(unittest.TestCase):
    def test_init(self):
        n_nodes = 100
        n_edges = 150
        n_nonedges = int(n_nodes ** 2 - n_nodes - n_edges * 2) / 2
        test_ratio = 0.1
        n_edges_test = int(test_ratio * n_edges)
        n_edges_test_neg = n_edges_test
        n_edges_train = n_edges - n_edges_test
        n_edges_train_neg = n_edges_train

        adata = simulate_spatial_adata(
            n_nodes = n_nodes,
            n_node_features = 0,
            n_edges = n_edges,
            random_seed = 1)

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
        self.assertEqual(n_edges_train_neg, dataset.n_edges_train_neg)
        self.assertEqual(n_edges_test, dataset.n_edges_test)
        self.assertEqual(n_edges_test_neg, dataset.n_edges_test_neg)
        self.assertEqual(
            int(dataset.A_train.sum()/2),
            len(dataset.edges_train))
        self.assertEqual(
            int(dataset.A_test.sum()/2),
             len(dataset.edges_test))
        self.assertEqual(
            int(dataset.A_train_diag.to_dense().sum()),
            int(dataset.A_train.sum() + dataset.n_nodes))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_train_neg, 
            dataset.edges_train))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_test_neg, 
            dataset.edges_test))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_test, 
            dataset.edges_train))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_test_neg, 
            dataset.edges_train_neg))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_test_neg, 
            dataset.edges_train))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_train_neg, 
            dataset.edges_test))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_train_neg, 
            dataset.edges))
        self.assertTrue(~has_overlapping_edges(
            dataset.edges_test_neg, 
            dataset.edges))
        self.assertEqual(
            dataset.A_train_diag.to_dense().sum(),
            dataset.n_nodes + dataset.n_edges_train * 2)
        for i, j in dataset.edges:
            self.assertEqual(dataset.A.toarray()[i, j], 1)
        for i, j in dataset.edges_train:
            self.assertEqual(dataset.A_train[i, j], 1)
        for i, j in dataset.edges_train:
            self.assertEqual(dataset.A_train_diag[i, j], 1)
        for i in range(dataset.n_nodes):
            self.assertEqual(dataset.A_train_diag[i, i], 1)
        for i, j in dataset.edges_train:
            self.assertNotEqual(dataset.A_train_diag_norm[i, j], 0)
        for i in range(dataset.n_nodes):
            self.assertNotEqual(dataset.A_train_diag_norm[i, i], 0)
        for i in range(dataset.n_nodes):
            self.assertNotEqual(dataset.A_train_diag_norm[i, i], 0)
            for j in range(dataset.n_nodes):
                if ((i, j) not in dataset.edges_train and (j, i) not in 
                dataset.edges_train and i != j):
                    self.assertEqual(dataset.A_train_diag_norm[i, j], 0)
        for i, j in dataset.edges_test:
            self.assertEqual(dataset.A_test[i, j], 1)   
        for i, j in dataset.edges_test_neg:
            self.assertEqual(dataset.A.toarray()[i, j], 0)
        self.assertIsInstance(dataset.A_train_diag, torch.Tensor)
        self.assertIsInstance(dataset.A_train_diag_norm, torch.Tensor)


if __name__ == '__main__':
    unittest.main()