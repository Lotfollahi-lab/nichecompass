"""
This module contains the functionality to compute all benchmarking metrics based
on the learned latent feature spaces of different deep generative models. The 
benchmark consists of metrics for spatial conservation as well as gene 
expression conservation.
"""

from typing import Optional

import mlflow
from anndata import AnnData

from autotalker.utils import compute_graph_connectivities
from .arclisi import compute_arclisi
from .cad import compute_cad
from .cca import compute_cca
from .gcd import compute_gcd
from .germse import compute_germse
from .mlnmi import compute_mlnmi
    

def compute_benchmarking_metrics(
        adata: Optional[AnnData]=None,
        spatial_model: bool=True,
        latent_key: str="",
        active_gp_names_key: str="",
        cell_type_key: str="cell-type",
        spatial_key: str="spatial",
        spatial_knng_key: str="autotalker_spatial_8nng",
        latent_knng_key: str="autotalker_latent_8nng",
        n_neighbors: int=8,
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
        Key under which the spatial nearest neighbor graph is / will be
        stored in ´adata.obsp´ with the suffix '_connectivities'.
    latent_knng_key:
        Key under which the latent nearest neighbor graph is / will be 
        stored in ´adata.obsp´ with the suffix '_connectivities'.
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
        ´gcd´, ´mlnmi´, ´cad´, ´arclisi´, ´germse´, ´cca´.
    """
    # Adding '_connectivities' as required by squidpy
    spatial_knng_connectivities_key = spatial_knng_key + "_connectivities"
    latent_knng_connectivities_key = latent_knng_key + "_connectivities"

    if spatial_model:
        if spatial_knng_connectivities_key not in adata.obsp:
            # Compute spatial (ground truth) connectivities
            adata.obsp[spatial_knng_connectivities_key] = (
                compute_graph_connectivities(
                    adata=adata,
                    feature_key=spatial_key,
                    n_neighbors=n_neighbors,
                    mode="knn",
                    seed=seed))

    if latent_knng_connectivities_key not in adata.obsp:
        # Compute latent connectivities
        adata.obsp[latent_knng_connectivities_key] = (
            compute_graph_connectivities(
                adata=adata,
                feature_key=latent_key,
                n_neighbors=n_neighbors,
                mode="knn",
                seed=seed))

    # Compute benchmarking metrics
    benchmark_dict = {}
    if spatial_model:
        benchmark_dict["gcd"] = compute_gcd(
            adata=adata,
            spatial_knng_key=spatial_knng_key,
            latent_knng_key=latent_knng_key)
        benchmark_dict["mlnmi"] = compute_mlnmi(
            adata=adata,
            spatial_knng_key=spatial_knng_key,
            latent_knng_key=latent_knng_key)
        benchmark_dict["cad"] = compute_cad(
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