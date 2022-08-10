import argparse

import numpy as np
import squidpy as sq

from autotalker.models import Autotalker
from autotalker.data import load_benchmark_spatial_adata
from autotalker.data import load_spatial_adata_from_csv
from autotalker.data import simulate_spatial_adata

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

    model = Autotalker(adata,
                       n_hidden = args.n_hidden,
                       n_latent = args.n_latent,
                       dropout_rate = args.dropout_rate)
    model.train()

if __name__ == '__main__':
    main(args)