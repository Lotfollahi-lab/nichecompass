import os

import matplotlib.pyplot as plt
import scanpy as sc


def plot_labeled_latent_adata(adata_latent,
                              save=False,
                              save_dir_path=None,
                              n_neighbors=50):
    show = True
    if save:
        show = False
        if save_dir_path is not None:
            os.makedirs(save_dir_path, exist_ok=True)
        else:
            raise ValueError("Please indicate ´save_dir_path´.")

    sc.pp.neighbors(adata_latent, n_neighbors=n_neighbors)
    sc.tl.umap(adata_latent)
    color = "cell_type" if adata_latent.obs["cell_type"] is not None else None
    print(color)
    print(adata_latent)
    sc.pl.umap(adata_latent,
               color=color,
               frameon=False,
               wspace=0.6,
               show=show)
    if save:
        plt.savefig(f"{save_dir_path}/labeled_latent_umap.png",
                    bbox_inches="tight")