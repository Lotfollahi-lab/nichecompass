import scanpy as sc

def plot_latent_space(
        self,
        show=True,
        save=False,
        dir_path=None,
        n_neighbors=8):
    if save:
        show=False
        if dir_path is None:
            save = False
    sc.pp.neighbors(self.adata_latent, n_neighbors=n_neighbors)
    sc.tl.umap(self.adata_latent)
    color = [
        'cell_type' if self.cell_type_names is not None else None,
        'batch' if self.batch_names is not None else None,
    ]
    sc.pl.umap(self.adata_latent,
               color=color,
               frameon=False,
               wspace=0.6,
               show=show)
    if save:
        plt.savefig(f'{dir_path}_batch.png', bbox_inches='tight')