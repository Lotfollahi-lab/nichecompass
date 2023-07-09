"""
This module contains the functionality to compute all benchmarking metrics based
on the spatial (physical) feature space and the learned latent feature space of
a deep generative model. The benchmark consists of metrics for spatial
conservation, biological conservation and batch correction.
"""

import time
from typing import Optional, Union

import mlflow
import scib_metrics
from anndata import AnnData

from ..utils import compute_knn_graph_connectivities_and_distances

from .cas import compute_cas
from .cca import compute_cca
from .clisis import compute_clisis
from .gcs import compute_gcs
from .gerr2 import compute_gerr2
from .mlami import compute_mlami
    

def compute_benchmarking_metrics(
        adata: AnnData,
        metrics: list=["gcs",
                       "mlami",
                       "cas",
                       "clisis",
                       "cari",
                       "cnmi",
                       "casw",
                       "clisi",
                       "cca",
                       "basw",
                       "bgc",
                       "bilisi",
                       "kbet"],
        cell_type_key: str="cell_type",
        batch_key: Optional[str]=None, 
        spatial_key: Optional[str]="spatial",
        latent_key: Optional[str]="nichecompass_latent",
        ger_genes: Optional[Union[str, list]]=None,
        n_jobs: int=1,
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
    ger_genes:
        Genes used for the gene expression reconstruction benchmark.
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
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
    start_time = time.time()
    
    # Compute nearest neighbor graphs
    if batch_key is None:
        print("Computing spatial nearest neighbor graphs for entire dataset...")
        for n_neighbors in [15, 50, 90]:
            compute_knn_graph_connectivities_and_distances(
                    adata=adata,
                    feature_key=spatial_key,
                    knng_key=f"nichecompass_spatial_{n_neighbors}knng",
                    n_neighbors=n_neighbors,
                    random_state=seed,
                    n_jobs=n_jobs)
        
        print("Computing latent nearest neighbor graphs for entire dataset...")
        for n_neighbors in [15, 50, 90]:
            compute_knn_graph_connectivities_and_distances(
                    adata=adata,
                    feature_key=latent_key,
                    knng_key=f"nichecompass_latent_{n_neighbors}knng",
                    n_neighbors=n_neighbors,
                    random_state=seed,
                    n_jobs=n_jobs)

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("Neighbor graphs computed. "
          f"Elapsed time: {minutes} minutes "
          f"{seconds} seconds.\n")

    # Compute benchmarking metrics
    benchmark_dict = {}
    
    # Spatial conservation metrics (unsupervised)
    if "gcs" in metrics:
        print("Computing GCS metric...")
        benchmark_dict["gcs"] = compute_gcs(
            adata=adata,
            batch_key=batch_key,
            spatial_knng_key="nichecompass_spatial_15knng",
            latent_knng_key="nichecompass_latent_15knng",
            seed=seed)

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("GCS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
        
    if "mlami" in metrics:
        print("Computing MLAMI Metric...")
        benchmark_dict["mlami"] = compute_mlami(
            adata=adata,
            batch_key=batch_key,
            spatial_knng_key="nichecompass_spatial_15knng",
            latent_knng_key="nichecompass_latent_15knng",
            seed=seed)
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("MLAMI metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
        
    # Spatial conservation metrics (cell-type-based)    
    if "cas" in metrics:
        print("Computing CAS metric...")
        benchmark_dict["cas"] = compute_cas(
            adata=adata,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
            spatial_knng_key="nichecompass_spatial_15knng",
            latent_knng_key="nichecompass_latent_15knng",
            seed=seed)
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CAS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
              
    if "clisis" in metrics:
        print("Computing CLISIS metric...")
        benchmark_dict["clisis"] = compute_clisis(
            adata=adata,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
            spatial_knng_key="nichecompass_spatial_90knng",
            latent_knng_key="nichecompass_latent_90knng",
            seed=seed)
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CLISIS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
    
    # Biological conservation metrics
    if "cnmi" in metrics or "cari" in metrics:
        print("Computing CNMI and CARI metrics...")
        cnmi_cari_dict = scib_metrics.nmi_ari_cluster_labels_kmeans(
            X=adata.obsm[latent_key],
            labels=adata.obs[cell_type_key])
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CNMI and CARI metrics computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

        if "cnmi" in metrics:
              benchmark_dict["cnmi"] = cnmi_cari_dict["nmi"]
        if "cari" in metrics:
              benchmark_dict["cari"] = cnmi_cari_dict["ari"]
              
    if "casw" in metrics:
        print("Computing CASW Metric...")
        benchmark_dict["casw"] = scib_metrics.silhouette_label(
            X=adata.obsm[latent_key],
            labels=adata.obs[cell_type_key])
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CASW metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
              
    if "clisi" in metrics:
        print("Computing CLISI metric...")
        benchmark_dict["clisi"] = scib_metrics.clisi_knn(
            X=adata.obsp["nichecompass_latent_90knng_distances"],
            labels=adata.obs[cell_type_key])
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CLISI metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
              
    if "gerr2" in metrics:
        print("Computing GERR2 Metric...")
        benchmark_dict["gerr2"] = compute_gerr2(
            adata=adata,
            latent_key=latent_key,
            regressor="mlp",
            selected_genes=ger_genes)
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("GERR2 metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
              
    if "cca" in metrics:
        print("Computing CCA Metric...")
        benchmark_dict["cca"] = compute_cca(
            adata=adata,
            cell_cat_key=cell_type_key,
            latent_key=latent_key,
            classifier="mlp",
            selected_cats=None)
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CCA metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

    # Batch correction metrics
    if batch_key is not None:
        if "basw" in metrics:
            print("Computing BASW Metric...")
            benchmark_dict["basw"] = scib_metrics.silhouette_batch(
                X=adata.obsm[latent_key],
                labels=adata.obs[cell_type_key],
                batch=adata.obs[batch_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("BASW metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
              
        if "bgc" in metrics:
            benchmark_dict["bgc"] = scib_metrics.graph_connectivity(
                X=adata.obsp["nichecompass_latent_15knng_distances"],
                labels=adata.obs[batch_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("BGC metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
              
        if "bilisi" in metrics:
            print("Computing BILISI Metric...")
            benchmark_dict["bilisi"] = scib_metrics.ilisi_knn(
                X=adata.obsp["nichecompass_latent_90knng_distances"],
                batches=adata.obs[batch_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("BILISI metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
              
        if "kbet" in metrics:
            benchmark_dict["kbet"] = scib_metrics.kbet_per_label(
                X=adata.obsp["nichecompass_latent_50knng_connectivities"],
                batches=adata.obs[batch_key],
                labels=adata.obs[cell_type_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("KBET metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.")

    # Track metrics with mlflow
    if mlflow_experiment_id is not None:
        for key, value in benchmark_dict.items():
            mlflow.log_metric(key, value)

    return benchmark_dict