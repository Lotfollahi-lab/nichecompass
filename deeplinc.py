

from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
from webbrowser import get

import numpy as np
import scipy.sparse as sp
import squidpy as sq
import torch
from torch import optim

from deeplinc.data import train_test_split, normalize_adj_mx
from deeplinc.nn import VGAE
from deeplinc.train import compute_gvae_loss, get_eval_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of units in hidden layer.')
parser.add_argument('--latent', type=int, default=16, help='Number of units in latent layer.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
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

    adj_mx_train, adj_mx_test, edges_train, edges_test, edges_test_neg = train_test_split(
        adata=adata,
        adj_mx_key="spatial_connectivities",
        test_ratio=0.1)

    print("Train test split completed...")
    print(adj_mx_train.shape)
    print(adj_mx_test.shape)
    print(edges_train.shape)
    print(edges_test.shape)
    print(edges_test_neg.shape)
    

    adj_mx_norm = normalize_adj_mx(adj_mx_train)
    adj_mx_labels = adj_mx_train + sp.eye(adj_mx_train.shape[0])
    adj_mx_labels = torch.FloatTensor(adj_mx_labels.toarray())

    # Reweight positive examples of edges (Aij = 1) in loss calculation using 
    # the proportion of negative examples relative to positive ones to achieve
    # equal total weighting of negative and positive examples
    n_neg_edges = n_nodes**2 - adj_mx_train.sum()
    n_pos_edges = adj_mx_train.sum()
    gvae_loss_pos_weight =  torch.FloatTensor([n_neg_edges / n_pos_edges])

    # Weighting of reconstruction loss compared to Kullback-Leibler divergence
    gvae_loss_norm_factor = n_nodes**2 / float(n_neg_edges * 2)

    model = VGAE(n_input, args.hidden, args.latent, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Start model training...")

    hidden_emb = None
    for epoch in range(args.n_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        adj_mx_pred, mu, logstd = model(features, adj_mx_norm)
        print(f"mu: {mu}")
        print(f"logstd: {logstd}")
        loss = compute_gvae_loss(
            preds=adj_mx_pred,
            labels=adj_mx_labels,
            mu=mu,
            logstd=logstd,
            n_nodes=n_nodes,
            norm_factor=gvae_loss_norm_factor,
            pos_weight=gvae_loss_pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(
            hidden_emb,
            adj_mx,
            edges_test,
            edges_test_neg)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    roc_score, ap_score, acc_score = get_eval_metrics(
        Z,
        edges_test,
        edges_test_neg,
        0.5)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)