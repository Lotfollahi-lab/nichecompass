import os
from typing import Optional

import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc


def plot_latent_umap(adata_latent: ad.AnnData,
                     show: bool=True,
                     save: bool=False,
                     save_dir_path: Optional[str]=None,
                     n_neighbors: int=50):
    """
    Plot latent umap from adata object.
    """
    if save:
        if save_dir_path is not None:
            os.makedirs(save_dir_path, exist_ok=True)
        else:
            raise ValueError("Please indicate ´save_dir_path´ or set ´save´ to "
                             "False.")

    sc.pp.neighbors(adata_latent, n_neighbors=n_neighbors)
    sc.tl.umap(adata_latent)
    color = "cell_type" if adata_latent.obs["cell_type"] is not None else None
    sc.pl.umap(adata_latent,
               color=color,
               frameon=False,
               wspace=0.6,
               show=show)
    
    if save:
        plt.savefig(f"{save_dir_path}/latent_umap.png", bbox_inches="tight")