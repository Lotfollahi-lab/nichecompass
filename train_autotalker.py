import argparse
import sys

import mlflow
import numpy as np
import squidpy as sq

from autotalker.data import load_spatial_adata_from_csv
from autotalker.models import Autotalker


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default="deeplinc_seqfish",
                    help="Dataset to use for model training.")
parser.add_argument("--n_epochs",
                    type=int,
                    default=10,
                    help="Number of epochs.")
parser.add_argument("--lr",
                    type=float,
                    default=0.01,
                    help="Initial learning rate.")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Batch size used for model training.")
parser.add_argument("--n_hidden",
                    type=int,
                    default=32,
                    help="Number of units in Autotalker VGAE hidden layer.")
parser.add_argument("--n_latent",
                    type=int,
                    default=16,
                    help="Number of units in Autotalker latent layer.")
parser.add_argument("--dropout_rate",
                    type=float,
                    default=0.,
                    help="Dropout rate (1 - keep probability).")
args = parser.parse_args()


def main(args):

    print(f"Using dataset {args.dataset}:")
    if args.dataset == "deeplinc_seqfish":
        adata = load_spatial_adata_from_csv("datasets/seqFISH/counts.csv",
                                            "datasets/seqFISH/adj.csv")
    elif args.dataset == "squidpy_seqfish":
        adata = sq.datasets.seqfish()
        sq.gr.spatial_neighbors(adata, radius = 0.04, coord_type="generic")
    elif args.dataset == "squidpy_slideseqv2":
        adata = sq.datasets.slideseqv2()
        sq.gr.spatial_neighbors(adata, radius = 30.0, coord_type="generic")

    print(f"Number of nodes: {adata.X.shape[0]}")
    print(f"Number of node features: {adata.X.shape[1]}")
    avg_edges_per_node = round(
        adata.obsp['spatial_connectivities'].toarray().sum(axis=0).mean(),2)
    print(f"Average number of edges per node: {avg_edges_per_node}")
    n_edges = int(np.triu(adata.obsp['spatial_connectivities'].toarray()).sum())
    print(f"Number of edges: {n_edges}", sep="")

    model = Autotalker(adata,
                       n_hidden=args.n_hidden,
                       n_latent=args.n_latent,
                       dropout_rate=args.dropout_rate)

    model.train(n_epochs=args.n_epochs,
                lr=args.lr,
                batch_size=args.batch_size)

    mlflow.log_param("dataset", args.dataset)

    model.save(dir_path="./model_artefacts",
               overwrite=True,
               save_adata=True,
               adata_file_name="adata.h5ad")
    
    model = Autotalker.load(dir_path="./model_artefacts",
                            adata=None,
                            adata_file_name="adata.h5ad",
                            use_cuda=True)

    print(model)
    print(model.is_trained_)

    print(model.get_latent_representation())


if __name__ == '__main__':
    main(args)