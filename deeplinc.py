

from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
from webbrowser import get
from deeplinc.data.utils import normalize_A

import numpy as np
import scipy.sparse as sp
import squidpy as sq
import torch
from torch import optim

from deeplinc.data import train_test_split, normalize_A
from deeplinc.nn import VGAE
from deeplinc.train import compute_vgae_loss, get_eval_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=250, help='Number of units in hidden layer.')
parser.add_argument('--latent', type=int, default=125, help='Number of units in latent layer.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

args = parser.parse_args()


def gae_for(args):
    print("Dataset...")
    adata = sq.datasets.visium_fluo_adata()
    sq.gr.spatial_neighbors(adata, n_rings=2, coord_type="grid", n_neighs=10)
    adj_mx = adata.obsp["spatial_connectivities"]
    features = torch.FloatTensor(adata.X.toarray())
    n_nodes = adj_mx.shape[0]
    n_input = features.size(1)
    #adj, features = load_data(args.dataset_str)
    #n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    #adj_orig = adj
    #adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    #adj_orig.eliminate_zeros()

    train_test_split_tuple = train_test_split(adata, "spatial_connectivities")
    A_train_nodiag, A_test_nodiag = train_test_split_tuple[:2]
    edges_train, edges_test_pos, edges_test_neg = train_test_split_tuple[2:]

    print("Train test split completed...")

    A_train_norm = normalize_A(A_train_nodiag)
    A_label_diag = A_train + sp.eye(A_train.shape[0])
    A_label_diag = torch.FloatTensor(A_label.toarray())

    # Reweight positive examples of edges (Aij = 1) in loss calculation using 
    # the proportion of negative examples relative to positive ones to achieve
    # equal total weighting of negative and positive examples
    n_neg_edges_train = n_nodes**2 - A_train.sum()
    n_pos_edges_train = A_train.sum()
    vgae_loss_pos_weight = torch.FloatTensor(
        [n_neg_edges_train / n_pos_edges_train])

    # Weighting of reconstruction loss compared to Kullback-Leibler divergence
    vgae_loss_norm_factor = n_nodes**2 / float(n_neg_edges_train * 2)

    model = VGAE(n_input, args.hidden, args.latent, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Start model training...")

    for epoch in range(args.n_epochs):
        start_time = time.time()
        model.train()
        A_rec_logits, mu, logstd = model(features, A_train_norm)
        #print(f"A_rec_probs: {A_rec_probs}")
        #print(f"mu: {mu}")
        #print(f"logstd: {logstd}")

        loss = compute_vgae_loss(
            A_rec_logits=A_rec_logits,
            A_label=A_label,
            mu=mu,
            logstd=logstd,
            n_nodes=n_nodes,
            norm_factor=vgae_loss_norm_factor,
            pos_weight=vgae_loss_pos_weight,
            debug=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        roc_score, ap_score, acc_score = get_eval_metrics(
        A_rec_logits,
        edges_test,
        edges_test_neg,
        0.5)

        print(f"Epoch: {epoch+1}")
        print(f"Train loss: {loss.item()}")
        print(f"Test roc score: {roc_score}")
        print(f"Test ap score: {ap_score}")
        print(f"Test accuracy: {acc_score}")
        print(f"Time: {time.time() - start_time}")
        print("--------------------")

    print("Optimization completed...")

    roc_score, ap_score, acc_score = get_eval_metrics(
        A_rec_logits,
        edges_test,
        edges_test_neg,
        0.5)

    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)