"""
This module contains the functionality to compute all benchmarking metrics based
on the physical (spatial) feature space and the learned latent feature spaces of
different deep generative models. The benchmark consists of metrics for spatial
conservation as well as linear recoverability of gene expression.
"""

from typing import Optional, Union

import mlflow
import scanpy as sc
from anndata import AnnData

from .cas import compute_cas
from .cca import compute_cca
from .clisis import compute_clisis
from .gcs import compute_gcs
from .gerr2 import compute_gerr2
from .mlami import compute_mlami
    

def compute_benchmarking_metrics(
        adata: AnnData,
        cell_type_key: str="cell_type",
        spatial_knng_key: str="nichecompass_spatial_knng",
        latent_knng_key: str="nichecompass_latent_knng",
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="nichecompass_latent",
        n_neighbors: Optional[int]=15, # sc.pp.neighbors default
        ger_genes: Optional[Union[str, list]]=None,
        seed: int=0,
        mlflow_experiment_id: Optional[str]=None) -> dict:
    """
    Parameters
    ----------
    adata:
        AnnData object to run the benchmarks for.
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    spatial_knng_key:
        Key under which the spatial nearest neighbor graph will be stored in
        ´adata.obsp´ with the suffix '_connectivities'.
    latent_knng_key:
        Key under which the latent nearest neighbor graph will be stored in
        ´adata.obsp´ with the suffix '_connectivities'.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from a model is stored in
        ´adata.obsm´.
    n_neighbors:
        Number of neighbors used for the construction of the nearest neighbor
        graphs from the spatial coordinates and the latent representation from a
        model.
    ger_genes:
        Genes used for the gene expression reconstruction benchmark.
    seed:
        Random seed for reproducibility.
    mlflow_experiment_id:
        ID of the Mlflow experiment used for tracking training parameters and
        metrics.

    Returns
    ----------
    benchmark_dict:
        Dictionary containing the calculated benchmarking metrics under keys
        ´gcs´, ´mlami´, ´cas´, ´clisis´, ´gerr2´ and ´cca´.
    """
    # Adding '_connectivities' as automatically added by sc.pp.neighbors
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_knng_connectivities_key not in adata.obsp:
        # Compute spatial (ground truth) connectivities
        sc.pp.neighbors(adata=adata,
                        use_rep=spatial_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=spatial_knng_key)

    if latent_knng_connectivities_key not in adata.obsp:
        # Compute latent connectivities
        sc.pp.neighbors(adata=adata,
                        use_rep=latent_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=latent_knng_key)

    # Compute benchmarking metrics
    benchmark_dict = {}
    print("Computing GCS...")
    benchmark_dict["gcs"] = compute_gcs(
        adata=adata,
        spatial_knng_key=spatial_knng_key,
        latent_knng_key=latent_knng_key,
        seed=seed)
    print("Computing MLAMI...")
    benchmark_dict["mlami"] = compute_mlami(
        adata=adata,
        spatial_knng_key=spatial_knng_key,
        latent_knng_key=latent_knng_key,
        seed=seed)
    print("Computing CAS...")
    benchmark_dict["cas"] = compute_cas(
        adata=adata,
        cell_type_key=cell_type_key,
        spatial_knng_key=spatial_knng_key,
        latent_knng_key=latent_knng_key,
        seed=seed)
    print("Computing CLISIS...")
    benchmark_dict["clisis"] = compute_clisis(
        adata=adata,
        cell_type_key=cell_type_key,
        spatial_knng_key=spatial_knng_key,
        latent_knng_key=latent_knng_key,
        seed=seed)
    print("Computing GERR2...")
    benchmark_dict["gerr2"] = compute_gerr2(
        adata=adata,
        latent_key=latent_key,
        regressor="mlp",
        selected_genes=ger_genes)
    print("Computing CCA...")
    benchmark_dict["cca"] = compute_cca(
        adata=adata,
        cell_cat_key=cell_type_key,
        latent_key=latent_key,
        classifier="mlp",
        selected_cats=None)

    # Track metrics with mlflow
    if mlflow_experiment_id is not None:
        for key, value in benchmark_dict.items():
            mlflow.log_metric(key, value)

    return benchmark_dict