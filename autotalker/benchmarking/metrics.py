"""
This module contains the functionality to compute all benchmarking metrics based
on the learned latent feature spaces of different deep generative models. The
benchmark consists of metrics for spatial conservation as well as gene
expression conservation.
"""

from typing import Optional

import mlflow
import scanpy as sc
from anndata import AnnData

from .arclisi import compute_arclisi
from .ctad import compute_ctad
from .cca import compute_cca
from .gcs import compute_gcs
from .germse import compute_germse
from .mlami import compute_mlami
from .rclisi import compute_rclisi
    

def compute_benchmarking_metrics(
        adata: AnnData,
        latent_key: str="autotalker_latent",
        active_gp_names_key: str="autotalker_active_gp_names",
        cell_type_key: str="cell_type",
        spatial_key: str="spatial",
        spatial_knng_key: str="autotalker_spatial_knng",
        latent_knng_key: str="autotalker_latent_knng",
        n_neighbors: int=15, # sc.pp.neighbors default
        seed: int=0,
        mlflow_experiment_id: Optional[str]=None) -> dict:
    """
    Parameters
    ----------
    adata:
        AnnData object to run the benchmarks for.
    latent_key:

    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    spatial_knng_key:
        Key under which the spatial nearest neighbor graph will be stored in
        ´adata.obsp´ with the suffix '_connectivities'.
    latent_knng_key:
        Key under which the latent nearest neighbor graph will be stored in
        ´adata.obsp´ with the suffix '_connectivities'.
    n_neighbors:
        Number of neighbors used for the construction of the nearest
        neighbor graphs from the spatial coordinates and the latent
        representation from the model.
    seed:
        Random seed for reproducibility.
    mlflow_experiment_id:
        ID of the Mlflow experiment used for tracking training parameters
        and metrics.

    Returns
    ----------
    benchmark_dict:
        Dictionary containing the calculated benchmarking metrics under keys
        ´gcs´, ´mlami´, ´ctad´, ´arclisi´, ´rclisi´, ´germse´ and ´cca´.
    """
    # Adding '_connectivities' as required by squidpy
    # spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    # latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_knng_key not in adata.uns:
        # Compute physical (ground truth) connectivities
        sc.pp.neighbors(adata=adata,
                        use_rep=spatial_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=spatial_knng_key)

    if latent_knng_key not in adata.uns:
        # Compute latent connectivities
        sc.pp.neighbors(adata=adata,
                        use_rep=latent_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=latent_knng_key)

    # Compute benchmarking metrics
    benchmark_dict = {}
    benchmark_dict["gcs"] = compute_gcs(
        adata=adata,
        spatial_knng_key=spatial_knng_key,
        latent_knng_key=latent_knng_key)
    benchmark_dict["mlami"] = compute_mlami(
        adata=adata,
        spatial_knng_key=spatial_knng_key,
        latent_knng_key=latent_knng_key)
    benchmark_dict["ctad"] = compute_ctad(
        adata=adata,
        cell_type_key=cell_type_key,
        spatial_knng_key=spatial_knng_key,
        latent_knng_key=latent_knng_key)
    benchmark_dict["arclisi"] = compute_arclisi(
        adata=adata,
        cell_type_key=cell_type_key,
        spatial_key=spatial_key,
        latent_key=latent_key,
        n_neighbors=n_neighbors,
        seed=seed)
    benchmark_dict["rclisi"] = compute_rclisi(
        adata=adata,
        cell_type_key=cell_type_key,
        spatial_knng_key=spatial_knng_key,
        latent_knng_key=latent_knng_key,
        knn_graph_n_neighbors=n_neighbors,
        seed=seed)
    benchmark_dict["germse"] = compute_germse(
        adata=adata,
        active_gp_names_key=active_gp_names_key,
        latent_key=latent_key,
        regressor="mlp",
        selected_gps=None,
        selected_genes=None)
    benchmark_dict["cca"] = compute_cca(
        adata=adata,
        cell_cat_key=cell_type_key,
        active_gp_names_key=active_gp_names_key,
        latent_key=latent_key,
        classifier="knn",
        selected_gps=None,
        selected_cats=None)

    # Track metrics with mlflow
    if mlflow_experiment_id is not None:
        for key, value in benchmark_dict.items():
            mlflow.log_metric(key, value)

    return benchmark_dict