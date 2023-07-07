"""
This module contains the functionality to compute all benchmarking metrics based
on the physical (spatial) feature space and the learned latent feature space of
a deep generative model. The benchmark consists of metrics for spatial
conservation, biological conservation and batch correction.
"""

import time
from typing import Optional, Union

import mlflow
import scanpy as sc
import scib_metrics
from anndata import AnnData

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
                       "gerr2",
                       "cca",
                       "basw",
                       "bgc",
                       "bilisi",
                       "kbet"],
        cell_type_key: str="cell_type",
        condition_key: Optional[str]=None,
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

    start_time = time.time()
    
    # Compute nearest neighbor graphs
    if spatial_knng_connectivities_key not in adata.obsp:
        print("Computing spatial nearest neighbor graph...")
        # Compute spatial neighbor graph
        sc.pp.neighbors(adata=adata,
                        use_rep=spatial_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=spatial_knng_key)
    else:
        print("Using precomputed spatial nearest neighbor graph...")

    if latent_knng_connectivities_key not in adata.obsp:
        print("Computing latent nearest neighbor graph...")
        # Compute latent neighbor graph
        sc.pp.neighbors(adata=adata,
                        use_rep=latent_key,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added=latent_knng_key)
    else:
        print("Using precomputed latent nearest neighbor graph...")              
        
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("Neighbor graphs computed. "
          f"Elapsed time: {minutes} minutes "
          f"{seconds} seconds./n")  

    # Compute benchmarking metrics
    benchmark_dict = {}
    
    # Spatial conservation metrics (unsupervised)
    if condition_key is None: # not for integrated data
        if "gcs" in metrics:
            print("Computing GCS metric...")
            benchmark_dict["gcs"] = compute_gcs(
                adata=adata,
                spatial_knng_key=spatial_knng_key,
                latent_knng_key=latent_knng_key,
                seed=seed)

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("GCS metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds./n")
            
        if "mlami" in metrics:
            print("Computing MLAMI Metric...")
            benchmark_dict["mlami"] = compute_mlami(
                adata=adata,
                condition_key=condition_key,
                spatial_knng_key=spatial_knng_key,
                latent_knng_key=latent_knng_key,
                seed=seed)
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("MLAMI metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds./n")
        
    # Spatial conservation metrics (cell-type-based)    
    if "cas" in metrics:
        print("Computing CAS metric...")
        benchmark_dict["cas"] = compute_cas(
            adata=adata,
            cell_type_key=cell_type_key,
            condition_key=condition_key,
            spatial_knng_key=spatial_knng_key,
            latent_knng_key=latent_knng_key,
            seed=seed)
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CAS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds./n")
              
    if "clisis" in metrics:
        print("Computing CLISIS metric...")
        benchmark_dict["clisis"] = compute_clisis(
            adata=adata,
            cell_type_key=cell_type_key,
            condition_key=condition_key,
            spatial_knng_key=spatial_knng_key,
            latent_knng_key=latent_knng_key,
            seed=seed)
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CLISIS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds./n")
    
    # Biological conservation metrics
    if "cari" in metrics or "cnmi" in metrics:
        print("Computing CNMI and CARI metrics...")
        cnmi, cari = scib_metrics.nmi_ari_cluster_labels_kmeans(
            X=adata.obsm[latent_key],
            labels=adata.obs[cell_type_key])
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CNMI and CARI metrics computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds./n")
              
        if "cari" in metrics:
              benchmark_dict["cari"] = cari
        if "cnmi" in metrics:
              benchmark_dict["cnmi"] = cnmi
              
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
              f"{seconds} seconds./n")
              
    if "clisi" in metrics:
        benchmark_dict["clisi"] = scib_metrics.clisi_knn(
            X=adata.obsm[latent_key],
            labels=adata.obs[cell_type_key])
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CLISI metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds./n")
              
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
              f"{seconds} seconds./n")
              
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
              f"{seconds} seconds./n")

    # Batch correction metrics
    if condition_key is not None:
        if "basw" in metrics:
            print("Computing BASW Metric...")
            benchmark_dict["basw"] = scib_metrics.silhouette_batch(
                X=adata.obsm[latent_key],
                labels=adata.obs[cell_type_key],
                batch=adata.obs[condition_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("BASW metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds./n")
              
        if "bgc" in metrics:
            benchmark_dict["bgc"] = scib_metrics.graph_connectivity(
                X=adata.obsm[latent_key],
                labels=adata.obs[condition_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("BGC metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds./n")
              
        if "bilisi" in metrics:
            print("Computing BILISI Metric...")
            benchmark_dict["bilisi"] = scib_metrics.ilisi_knn(
                X=adata.obsm[latent_key],
                batches=adata.obs[condition_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("BILISI metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds./n")
              
        if "kbet" in metrics:
            benchmark_dict["kbet"] = scib_metrics.kbet_per_label(
                X=adata.obsm[latent_key],
                batches=adata.obs[condition_key],
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