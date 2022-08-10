import argparse
import sys
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import squidpy as sq
import torch

from autotalker.data import SpatialAnnDataPyGDataset
from autotalker.data import load_benchmark_spatial_adata
from autotalker.data import load_spatial_adata_from_csv
from autotalker.data import simulate_spatial_adata
from autotalker.modules import VGAE
from autotalker.train import compute_vgae_loss_parameters
from autotalker.train import compute_vgae_loss
from autotalker.train import plot_loss
from autotalker.train import get_eval_metrics
from autotalker.train import plot_eval_metrics
from autotalker.train import prepare_data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type = str,
    default = "MERFISH",
    help = "Dataset to use for model training.")
parser.add_argument(
    "--n_epochs",
    type = int,
    default = 200,
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
    default = 0.01, # 0.0004 for visium
    help = "Initial learning rate.")
parser.add_argument(
    "--dropout_rate",
    type = float,
    default = 0.,
    help = "Dropout rate (1 - keep probability).")
args = parser.parse_args()


def main(args):

    mlflow.set_experiment("autotalker")

    # Logging hyperparameters
    mlflow.log_param("dataset", args.dataset)
    mlflow.log_param("n_epochs", args.n_epochs)
    mlflow.log_param("n_hidden", args.n_hidden)
    mlflow.log_param("n_latent", args.n_latent)
    mlflow.log_param("lr", args.lr)
    mlflow.log_param("dropout_rate", args.dropout_rate)

    # Configure PyTorch GPU device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Configured device {device}...")
    
    print("Loading dataset:")

    if args.dataset == "cora":
        print("cora")
        adata = load_benchmark_spatial_adata("cora")
    elif args.dataset == "citeseer":
        print("citeseer")
        adata = load_benchmark_spatial_adata("citeseer")
    elif args.dataset == "pubmed":
        print("pubmed")
        adata = load_benchmark_spatial_adata("pubmed")
    elif args.dataset == "seqFISH":
        print("seqFISH")
        adata = load_spatial_adata_from_csv(
            "datasets/seqFISH/counts.csv",
            "datasets/seqFISH/adj.csv")
    elif args.dataset == "MERFISH":
        print("MERFISH")
        adata = load_spatial_adata_from_csv(
            "datasets/MERFISH/counts.csv",
            "datasets/MERFISH/adj.csv")
    elif args.dataset == "visium":
        print("visium")
        adata = sq.datasets.visium_fluo_adata()
        sq.gr.spatial_neighbors(
            adata,
            n_rings = 2,
            coord_type = "grid",
            n_neighs = 6)
    elif args.dataset == "simulation":
        print("simulation")
        adata = simulate_spatial_adata(
            n_nodes = 1600,
            n_node_features = 200,
            n_edges = 3000,
            adj_nodes_feature_multiplier = 5)

    print(f"Number of nodes: {adata.X.shape[0]}")
    print(f"Number of node features: {adata.X.shape[1]}")
    print(
        "Number of edges: ", 
        f"{int(np.triu(adata.obsp['spatial_connectivities'].toarray()).sum())}",
        sep="")
    # print(adata.X)
    # print(adata.obsp["spatial_connectivities"].toarray())
    # sys.exit(1)

    print("Initializing and preprocessing dataset...")

    train_data, val_data, test_data, train_adj_labels = prepare_data(adata)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    train_adj_labels = train_adj_labels.to(device)
    print(train_data)

    print("Dataset initialized and preprocessed...")

    print("Calculating VGAE loss parameters:")

    vgae_loss_norm_factor, vgae_loss_pos_weight = compute_vgae_loss_parameters(
        train_adj_labels)

    # vgae_loss_norm_factor = 200

    print(f"VGAE loss pos weight: {vgae_loss_pos_weight.item()}")
    print(f"VGAE loss norm factor: {vgae_loss_norm_factor}")

    vgae_loss_pos_weight = vgae_loss_pos_weight.to(device)

    print("Initializing model...")
    
    model = VGAE(
        train_data.x.size(1),
        args.n_hidden,
        args.n_latent,
        args.dropout_rate)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    print("--------------------")
    print("--------------------")
    print("Starting model training...")

    start_time = time.time()

    auroc_scores_val = []
    auprc_scores_val = []
    best_acc_scores_val = []
    best_f1_scores_val = []
    losses = []

    for epoch in range(args.n_epochs):
 
        adj_rec_logits, mu, logstd = model(train_data.x, train_data.edge_index)
        # A_rec_logits, mu, logstd = model(X, A_norm)

        loss = compute_vgae_loss(
            A_rec_logits = adj_rec_logits,
            A_label = train_adj_labels,
            mu = mu,
            logstd = logstd,
            n_nodes = train_data.x.size(0),
            norm_factor = vgae_loss_norm_factor,
            pos_weight = vgae_loss_pos_weight,
            debug = False)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            A_rec_logits_mu = torch.mm(mu, mu.t())
            A_rec_probs = torch.sigmoid(A_rec_logits_mu)

            auroc_score_val, auprc_score_val, best_acc_score_val, best_f1_score_val  = get_eval_metrics(
                A_rec_probs,
                val_data.pos_edge_label_index,
                val_data.neg_edge_label_index)

            auroc_scores_val.append(auroc_score_val)
            auprc_scores_val.append(auprc_score_val)
            best_acc_scores_val.append(best_acc_score_val)
            best_f1_scores_val.append(best_f1_score_val)

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print("--------------------")
                print(f"Epoch: {epoch+1}")
                print(f"Train loss: {loss.item()}")
                print(f"Val (balanced) AUROC score: {auroc_score_val}")
                print(f"Val (balanced) AUPRC score: {auprc_score_val}")
                print(f"Val (balanced) best ACC score: {best_acc_score_val}")
                print(f"Val (balanced) best F1 score: {best_f1_score_val}")
                print(f"Elapsed training time: {time.time() - start_time}")
    
    print("--------------------")
    print("Model training finished...")

    eval_metrics_val = {"auroc": auroc_scores_val,
                        "auprc": auprc_scores_val,
                        "best_acc": best_acc_scores_val,
                        "best_f1": best_f1_scores_val}

    plot_loss(losses)
    mlflow.log_artifact("images/training_loss.png")
    plot_eval_metrics(eval_metrics_val)               
    mlflow.log_artifact("images/eval_metrics.png")

    auroc_score_test, auprc_score_test, best_acc_score_test, best_f1_score_test = get_eval_metrics(
        A_rec_probs,
        test_data.pos_edge_label_index,
        test_data.neg_edge_label_index)
    print(f"Test (balanced) AUROC score: {auroc_score_test}")
    print(f"Test (balanced) AUPRC score: {auprc_score_test}")
    print(f"Test (balanced) best ACC score: {best_acc_score_test}")
    print(f"Test (balanced) best F1 score: {best_f1_score_test}")

    mlflow.log_metric("auroc_score_test", auroc_score_test)
    mlflow.log_metric("auprc_score_test", auprc_score_test)
    mlflow.log_metric("best_acc_score_test", best_acc_score_test)
    mlflow.log_metric("best_f1_score_test", best_f1_score_test)


if __name__ == '__main__':
    main(args)