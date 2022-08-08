import argparse
import sys
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import squidpy as sq
import torch

from autotalker.data import SpatialAnnDataDataset
from autotalker.data import load_benchmark_spatial_adata
from autotalker.data import load_spatial_adata_from_csv
from autotalker.data import simulate_spatial_adata
from autotalker.nn import VGAE
from autotalker.train import compute_vgae_loss_parameters
from autotalker.train import compute_vgae_loss
from autotalker.train import get_eval_metrics


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type = str,
    default = "simulation",
    help = "Dataset to use for model training.")
parser.add_argument(
    "--seed",
    type = int,
    default = 1,
    help = "Random seed.")
parser.add_argument(
    "--n_epochs",
    type = int,
    default = 200, # 200
    help = "Number of epochs to train.")
parser.add_argument(
    "--n_hidden",
    type = int,
    default = 32, # 250
    help = "Number of units in VGAE hidden layer.")
parser.add_argument(
    "--n_latent",
    type = int,
    default = 16, # 125
    help = "Number of units in VGAE latent layer.")
parser.add_argument(
    "--lr",
    type = float,
    default = 0.01, # 0.0004
    help = "Initial learning rate.")
parser.add_argument(
    "--dropout",
    type = float,
    default = 0.,
    help = "Dropout rate (1 - keep probability).")
args = parser.parse_args()


def main(args):
    
    # adata = sq.datasets.visium_fluo_adata()
    # sq.gr.spatial_neighbors(adata, n_rings=2, coord_type="grid", n_neighs=10)
    # A = adata.obsp["spatial_connectivities"]
    # X = torch.FloatTensor(adata.X.toarray())
    # n_nodes = A.shape[0]
    # n_input = X.size(1)
    #adj, features = load_data(args.dataset_str)
    #n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    #adj_orig = adj
    #adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    #adj_orig.eliminate_zeros()

    # np.random.seed(1)
    # n_nodes = 100
    # node_dim = 100
    # n_edges = 150
    # n_nonedges = int(n_nodes ** 2 - n_nodes - n_edges * 2) / 2
    # test_ratio = 0.1
    # n_edges_test = int(test_ratio * n_edges)
    # n_edges_train = n_edges - n_edges_test
    # n_edges_test_neg = int(test_ratio * n_edges)
    # # Identity feature matrix
    # X = np.eye(n_nodes, node_dim).astype("float32")
    # print(f"X:\n {X}", "\n")
    # 
    # # Symmetric adjacency matrix
    # A = np.random.rand(n_nodes, n_nodes)
    # A = (A + A.T)/2
    # np.fill_diagonal(A, 0)
    # threshold = np.sort(A, axis = None)[-n_edges*2]
    # A = (A >= threshold).astype("int")
    # print(f"A:\n {A}", "\n")
# 
    # adata = ad.AnnData(X)
    # adata.obsp["spatial_connectivities"] = sp.csr_matrix(A)

    # Configure PyTorch GPU device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Configuring device {device} ...")
    
    print("Loading data...")

    if args.dataset == "cora":
        adata = load_benchmark_spatial_adata("cora")
    elif args.dataset == "citeseer":
        adata = load_benchmark_spatial_adata("citeseer")
    elif args.dataset == "pubmed":
        adata = load_benchmark_spatial_adata("pubmed")
    elif args.dataset == "seqFISH":
        adata = load_spatial_adata_from_csv(
            "datasets/seqFISH/counts.csv",
            "datasets/seqFISH/adj.csv")
    elif args.dataset == "simulation":
        adata = simulate_spatial_adata(
            n_nodes = 8,
            n_node_features = 0,
            n_edges = 10,
            adj_nodes_feature_multiplier = 10)


    print(adata.X)
    #print(adata.X.shape)
    print(adata.obsp["spatial_connectivities"].toarray())
    #sys.exit(1)

    # adata = spatial_adata_from_csv()

    print("Initializing and preprocessing dataset...")

    dataset = SpatialAnnDataDataset(
        adata,
        A_key = "spatial_connectivities",
        test_ratio = 0.1)

    X = dataset.X.to(device)
    A_label = dataset.A_train_diag.to(device)
    A_norm = dataset.A_train_diag_norm.to(device)

    print("Dataset initialized and preprocessed...")

    #A_train_norm = normalize_A(A_train_nodiag)
    #A_label_diag = A_train + sp.eye(A_train.shape[0])
    #A_label_diag = torch.FloatTensor(A_label.toarray())

    print("Calculating VGAE loss parameters:")

    vgae_loss_norm_factor, vgae_loss_pos_weight = compute_vgae_loss_parameters(
        A_label)

    # vgae_loss_norm_factor = 200

    print(f"VGAE loss pos weight: {vgae_loss_pos_weight}")
    print(f"VGAE loss norm factor: {vgae_loss_norm_factor}")
    
    model = VGAE(
        dataset.n_node_features,
        args.n_hidden,
        args.n_latent,
        args.dropout)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    vgae_loss_pos_weight = vgae_loss_pos_weight.to(device)

    print("Starting model training...")

    for epoch in range(args.n_epochs):
        start_time = time.time()
 
        A_rec_logits, mu, logstd = model(X, A_norm)

        loss = compute_vgae_loss(
            A_rec_logits = A_rec_logits,
            A_label = A_label.to_dense(),
            mu = mu,
            logstd = logstd,
            n_nodes = dataset.n_nodes,
            norm_factor = vgae_loss_norm_factor,
            pos_weight = vgae_loss_pos_weight,
            debug = False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            A_rec_logits_mu = torch.mm(mu, mu.t())
            A_rec_probs = torch.sigmoid(A_rec_logits_mu)

            eval_metrics_train = get_eval_metrics(
                A_rec_probs,
                dataset.edges_train,
                dataset.edges_train_neg)

            auroc_score_train = eval_metrics_train[0]
            auprc_score_train = eval_metrics_train[1]
            acc_score_train = eval_metrics_train[2]
            f1_score_train = eval_metrics_train[3]
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print("--------------------")
                print(f"Epoch: {epoch+1}")
                print(f"Train loss: {loss.item()}")
                print(f"Train (balanced) AUROC score: {auroc_score_train}")
                print(f"Train (balanced) AUPRC score: {auprc_score_train}")
                print(f"Train (balanced) ACC score: {acc_score_train}")
                print(f"Train (balanced) F1 score: {f1_score_train}")
                print(f"Elapsed training time: {time.time() - start_time}")
    
    print("--------------------")
    print("Model training finished...")

    eval_metrics_test = get_eval_metrics(
        A_rec_probs,
        dataset.edges_test,
        dataset.edges_test_neg)
    auroc_score_test = eval_metrics_test[0]
    auprc_score_test = eval_metrics_test[1]
    acc_score_test = eval_metrics_test[2]
    f1_score_test = eval_metrics_test[3]
    print(f"Test (balanced) AUROC score: {auroc_score_test}")
    print(f"Test (balanced) AUPRC score: {auprc_score_test}")
    print(f"Test (balanced) ACC score: {acc_score_test}")
    print(f"Test (balanced) F1 score: {f1_score_test}")


if __name__ == '__main__':
    main(args)