"""
This module contains the functionality to compute all benchmarking metrics based
on the spatial (physical) feature space and the learned latent feature space of
a deep generative model. The benchmark consists of metrics for spatial
conservation, biological conservation, niche identification and batch
correction.
"""

import time
from typing import Optional, Union
import gc

import mlflow
import numpy as np
import scanpy as sc
import scib_metrics
from anndata import AnnData

from .utils import compute_knn_graph_connectivities_and_distances

from .cas import compute_cas
from .clisis import compute_clisis
from .gcs import compute_gcs
from .mlami import compute_mlami
from .nasw import compute_nasw
    

def compute_benchmarking_metrics(
        adata: AnnData,
        metrics: list=["cas", # global spatial conservation (cell label supervised)
                       "mlami", # global spatial conservation (unsupervised)
                       "clisis", # local spatial conservation (cell label supervised)
                       "gcs", # local spatial conservation (unsupervised)
                       "cnmi", # niche coherence
                       "nasw", # niche coherence
                       "basw", # batch correction
                       "blisi", # batch correction
                       "kbet", # batch correction
                       "pcr"], # batch correction
        cell_type_key: str="cell_type",
        batch_key: Optional[str]=None, 
        spatial_key: str="spatial",
        latent_key: str="nichecompass_latent",
        pcr_X_pre: Optional[np.array]=None,
        n_jobs: int=1,
        seed: int=0,
        mlflow_experiment_id: Optional[str]=None) -> dict:
    """
    Compute all specified benchmarking metrics.

    Parameters
    ----------
    adata:
        AnnData object to run the benchmarks for.
    metrics:
        List of metrics which will be computed.
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_key:
        Key under which the latent representation from a model is stored in
        ´adata.obsm´.
    pcr_X_pre:
        The unintegrated feature space for the computation of the pcr metric.
        If None, computes PCA on the raw counts stored in ´adata.X´.
    n_jobs:
        Number of jobs to use for parallelization of neighbor search.
    seed:
        Random seed for reproducibility.
    mlflow_experiment_id:
        ID of the Mlflow experiment used for tracking metrics.

    Returns
    ----------
    benchmarking_dict:
        Dictionary containing the calculated benchmarking metrics.
    """
    start_time = time.time()

    # Metrics use different k's for the knn graph
    # Based on specified metrics, determine which knn graphs to compute
    n_neighbors_list = []
    if any(metric in ["gcs", "mlami", "cas", "nasw"] for metric in
           metrics):
        n_neighbors_list.append(15) # default k for connectivity-based
                                    # metrics
    if any(metric in ["kbet"] for metric in metrics):
        n_neighbors_list.append(50) # kbet-specific k
    if any(metric in ["clisis", "clisi", "blisi"] for metric in metrics):
        n_neighbors_list.append(90) # lisi-specific k
    
    benchmarking_dict = {}
    
    # Compute nearest neighbor graphs
    # Otherwise different metrics require different neighbor graphs and
    # this will be handled in the metric functions themselves
    if len(n_neighbors_list) > 0:
        # Compute spatial nearest neighbor graphs
        for n_neighbors in n_neighbors_list:
            if (f"{spatial_key}_{n_neighbors}knng_connectivities"
                not in adata.obsp):
                print("Computing spatial nearest neighbor graph with "
                      f"{n_neighbors} neighbors for entire dataset...")
                compute_knn_graph_connectivities_and_distances(
                        adata=adata,
                        feature_key=spatial_key,
                        knng_key=f"{spatial_key}_{n_neighbors}knng",
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        n_jobs=n_jobs)
            else:
                print(f"Using precomputed spatial nearest neighbor graph "
                      f"with {n_neighbors} neighbors...")

        # Compute latent nearest neighbor graphs
        for n_neighbors in n_neighbors_list:
            if (f"{latent_key}_{n_neighbors}knng_connectivities"
                not in adata.obsp):
                print("Computing latent nearest neighbor graph with "
                      f"{n_neighbors} neighbors for entire dataset...")
                compute_knn_graph_connectivities_and_distances(
                        adata=adata,
                        feature_key=latent_key,
                        knng_key=f"{latent_key}_{n_neighbors}knng",
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        n_jobs=n_jobs) # pynndescent has to be version 0.5.8 
                                       # otherwise this can throw errors for some random seeds and big latents
            else:
                print(f"Using precomputed latent nearest neighbor graph "
                      f"with {n_neighbors} neighbors...")

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("Neighbor graphs computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
            
    # Compute benchmarking metrics
    print("Computing benchmarking metrics...")
    
    # Global spatial conservation metric (cell label supervised)    
    if "cas" in metrics:
        print("Computing CAS metric...")
        benchmarking_dict["cas"] = compute_cas(
            adata=adata,
            cell_type_key=cell_type_key,
            batch_key=batch_key,
            spatial_knng_key=f"{spatial_key}_15knng",
            latent_knng_key=f"{latent_key}_15knng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            seed=seed)
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CAS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

    # Global spatial conservation metric (unsupervised)
    if "mlami" in metrics:
        print("Computing MLAMI Metric...")
        benchmarking_dict["mlami"] = compute_mlami(
            adata=adata,
            batch_key=batch_key,
            spatial_knng_key=f"{spatial_key}_15knng",
            latent_knng_key=f"{latent_key}_15knng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            seed=seed)
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("MLAMI metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
    
    # Local spatial conservation metric (cell label supervised)
    if "clisis" in metrics:
        try:
            print("Computing CLISIS metric...")
            benchmarking_dict["clisis"] = compute_clisis(
                adata=adata,
                cell_type_key=cell_type_key,
                batch_key=batch_key,
                spatial_knng_key=f"{spatial_key}_90knng",
                latent_knng_key=f"{latent_key}_90knng",
                spatial_key=spatial_key,
                latent_key=latent_key,
                seed=seed)

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("CLISIS metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
        except:
            print("Could not compute CLISIS metric.")
            benchmarking_dict["clisis"] = 0.
    
    # Local spatial conservation metric (unsupervised) 
    if "gcs" in metrics:
        print("Computing GCS metric...")
        benchmarking_dict["gcs"] = compute_gcs(
            adata=adata,
            batch_key=batch_key,
            spatial_knng_key=f"{spatial_key}_15knng",
            latent_knng_key=f"{latent_key}_15knng",
            spatial_key=spatial_key,
            latent_key=latent_key,
            seed=seed)

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("GCS metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")            
    
    # Niche coherence metrics
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
              benchmarking_dict["cnmi"] = cnmi_cari_dict["nmi"]
        if "cari" in metrics:
              benchmarking_dict["cari"] = cnmi_cari_dict["ari"]
    
    if "casw" in metrics:
        print("Computing CASW Metric...")
        benchmarking_dict["casw"] = scib_metrics.silhouette_label(
            X=adata.obsm[latent_key],
            labels=adata.obs[cell_type_key])
              
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("CASW metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")
    
    if "clisi" in metrics:
        try:
            print("Computing CLISI metric...")
            benchmarking_dict["clisi"] = scib_metrics.clisi_knn(
                X=adata.obsp[f"{latent_key}_90knng_distances"],
                labels=adata.obs[cell_type_key])

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("CLISI metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
        except:
            print("Could not compute CLISI metric.")
            benchmarking_dict["clisi"] = 0.
        
    if "nasw" in metrics:
        print("Computing NASW Metric...")
        benchmarking_dict["nasw"] = compute_nasw(
                adata=adata,
                latent_knng_key=f"{latent_key}_15knng",
                latent_key=latent_key,
                seed=seed)
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("NASW metric computed. "
              f"Elapsed time: {minutes} minutes "
              f"{seconds} seconds.\n")

    # Batch correction metrics
    if batch_key is not None:
        if "basw" in metrics:
            print("Computing BASW Metric...")
            benchmarking_dict["basw"] = scib_metrics.silhouette_batch(
                X=adata.obsm[latent_key],
                labels=adata.obs[cell_type_key],
                batch=adata.obs[batch_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("BASW metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.\n")
              
        if "blisi" in metrics:
            try:
                print("Computing BLISI Metric...")
                benchmarking_dict["blisi"] = scib_metrics.ilisi_knn(
                    X=adata.obsp[f"{latent_key}_90knng_distances"],
                    batches=adata.obs[batch_key])

                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                print("BLISI metric computed. "
                      f"Elapsed time: {minutes} minutes "
                      f"{seconds} seconds.\n")
            except:
                print("Could not compute BLISI metric.")
                benchmarking_dict["blisi"] = 0.
              
        if "kbet" in metrics:
            benchmarking_dict["kbet"] = scib_metrics.kbet_per_label(
                X=adata.obsp[f"{latent_key}_50knng_connectivities"],
                batches=adata.obs[batch_key],
                labels=adata.obs[cell_type_key])
              
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("KBET metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.")
            
        if "pcr" in metrics:
            # https://github.com/yoseflab/scib-metrics/blob/0.4.0/src/scib_metrics/benchmark/_core.py#L171
            if pcr_X_pre is None:
                if "X_pca" not in adata.obsm:
                    sc.tl.pca(adata, use_highly_variable=False)
                pcr_X_pre = adata.obsm["X_pca"]
            benchmarking_dict["pcr"] = scib_metrics.pcr_comparison(
                X_pre=pcr_X_pre,
                X_post=adata.obsm[latent_key],
                covariate=adata.obs[batch_key],
                categorical=True)
            
            print(benchmarking_dict["pcr"])
            
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print("PCR metric computed. "
                  f"Elapsed time: {minutes} minutes "
                  f"{seconds} seconds.")

    # Track metrics with mlflow
    if mlflow_experiment_id is not None:
        for key, value in benchmarking_dict.items():
            mlflow.log_metric(key, value)

    return benchmarking_dict
