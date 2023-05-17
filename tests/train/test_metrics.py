import numpy as np
import pytest
import torch

from nichecompass.train import eval_metrics
from nichecompass.data import simulate_spatial_adata
from nichecompass.data import SpatialAnnDataDataset


def test_eval_metrics():
    adata = simulate_spatial_adata(
        n_nodes = 100,
        n_node_features = 0,
        n_edges = 150,
        random_seed = 1)
    
    dataset = SpatialAnnDataDataset(adata)
    
    A_rec_logits = torch.tensor(dataset.A_train.toarray())
    
    auroc_score, auprc_score, acc_score, f1_score = eval_metrics(
        A_rec_logits = A_rec_logits,
        edges_pos = dataset.edges_train,
        edges_neg = dataset.edges_train_neg)
    
    assert auroc_score == 1
    assert auprc_score == 1
    assert acc_score == 1
    assert f1_score == 1

    auroc_score, auprc_score, acc_score, f1_score = eval_metrics(
        A_rec_logits = A_rec_logits,
        edges_pos = dataset.edges_train_neg,
        edges_neg = dataset.edges_train,
        debug = True)

    assert acc_score == 0.5
    
    A_rec_logits = torch.tensor(dataset.A_test.toarray())

    auroc_score, auprc_score, acc_score, f1_score = eval_metrics(
        A_rec_logits = A_rec_logits,
        edges_pos = dataset.edges_test,
        edges_neg = dataset.edges_test_neg)
    
    assert auroc_score == 1
    assert auprc_score == 1
    assert acc_score == 1
    assert f1_score == 1

    auroc_score, auprc_score, acc_score, f1_score = eval_metrics(
        A_rec_logits = A_rec_logits,
        edges_pos = dataset.edges_test_neg,
        edges_neg = dataset.edges_test)

    assert acc_score == 0.5

