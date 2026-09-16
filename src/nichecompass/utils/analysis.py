"""
This module contains utilities to analyze niches inferred by the NicheCompass
model.
"""

import warnings
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from anndata import AnnData
from matplotlib import cm, colors
from matplotlib.lines import Line2D
import networkx as nx

from ..models import NicheCompass


def aggregate_obsp_matrix_per_cell_type(
        adata: AnnData,
        obsp_key: str,
        cell_type_key: str="cell_type",
        group_key: Optional[str]=None,
        agg_rows: bool=False):
    """
    Generic function to aggregate adjacency matrices stored in
    ´adata.obsp[obsp_key]´ on cell type level. It can be used to aggregate the
    node label aggregator aggregation weights alpha or the reconstructed adjacency
    matrix of a trained NicheCompass model by neighbor cell type for downstream
    analysis.

    Parameters
    ----------
    adata:
        AnnData object which contains outputs of NicheCompass model training.
    obsp_key:
        Key in ´adata.obsp´ where the matrix to be aggregated is stored.
    cell_type_key:
        Key in ´adata.obs´ where the cell type labels are stored.
    group_key:
        Key in ´adata.obs´ where additional grouping labels are stored.    
    agg_rows:
        If ´True´, also aggregate over the observations on cell type level.

    Returns
    ----------
    cell_type_agg_df:
        Pandas DataFrame with the aggregated obsp values (dim: n_obs x
        n_cell_types if ´agg_rows == False´, else n_cell_types x n_cell_types).
    """
    n_obs = len(adata)
    n_cell_types = adata.obs[cell_type_key].nunique()
    sorted_cell_types = sorted(adata.obs[cell_type_key].unique().tolist())

    cell_type_label_encoder = {k: v for k, v in zip(
        sorted_cell_types,
        range(n_cell_types))}

    # Retrieve non zero indices and non zero values, and create row-wise
    # observation cell type index
    nz_obsp_idx = adata.obsp[obsp_key].nonzero()
    neighbor_cell_type_index = adata.obs[cell_type_key].iloc[nz_obsp_idx[1]].map(
        cell_type_label_encoder).values
    adata.obsp[obsp_key].eliminate_zeros() # In some sparse reps 0s can appear
    nz_obsp = adata.obsp[obsp_key].data

    # Use non zero indices, non zero values and row-wise observation cell type
    # index to construct new df with cell types as columns and row-wise
    # aggregated values per cell type index as values
    cell_type_agg = np.zeros((n_obs, n_cell_types))
    np.add.at(cell_type_agg,
              (nz_obsp_idx[0], neighbor_cell_type_index),
              nz_obsp)
    cell_type_agg_df = pd.DataFrame(
        cell_type_agg,
        columns=sorted_cell_types)
    
    # Add cell type labels of observations
    cell_type_agg_df[cell_type_key] = adata.obs[cell_type_key].values

    # If specified, add group label
    if group_key is not None:
        cell_type_agg_df[group_key] = adata.obs[group_key].values

    if agg_rows:
        # In addition, aggregate values across rows to get a
        # (n_cell_types x n_cell_types) df
        if group_key is not None:
            cell_type_agg_df = cell_type_agg_df.groupby(
                [group_key, cell_type_key]).sum()
        else:
            cell_type_agg_df = cell_type_agg_df.groupby(cell_type_key).sum()

        # Sort index to have same order as columns
        cell_type_agg_df = cell_type_agg_df.loc[
            sorted(cell_type_agg_df.index.tolist()), :]
        
    return cell_type_agg_df


def create_cell_type_chord_plot_from_df(
        adata: AnnData,
        df: pd.DataFrame,
        link_threshold: float=0.01,
        cell_type_key: str="cell_type",
        group_key: Optional[str]=None,
        groups: str="all",
        plot_label: str="Niche",
        save_fig: bool=False,
        file_path: Optional[str]=None):
    """
    Create a cell type chord diagram per group based on an input DataFrame.

    Parameters
    ----------
    adata:
        AnnData object which contains outputs of NicheCompass model training.
    df:
        A Pandas DataFrame that contains the connection values for the chord
        plot (dim: (n_groups x n_cell_types) x n_cell_types).
    link_threshold:
        Ratio of link strength that a cell type pair needs to exceed compared to
        the cell type pair with the maximum link strength to be considered a
        link for the chord plot.
    cell_type_key:
        Key in ´adata.obs´ where the cell type labels are stored.
    group_key:
        Key in ´adata.obs´ where additional group labels are stored.
    groups:
        List of groups that will be plotted. If ´all´, plot all groups.
    plot_label:
        Shared label for the plots.
    save_fig:
        If ´True´, save the figure.
    file_path:
        Path where to save the figure.
    """
    try:
        import holoviews as hv
    except ImportError as error:
        # The module level import was commented out, which left every ´hv´
        # reference in this function unbound: the first statement raised
        # ´NameError: name 'hv' is not defined´ rather than saying what to
        # install. holoviews is an optional dependency, so it stays out of the
        # module imports and out of ´pyproject.toml´.
        raise ImportError(
            "create_cell_type_chord_plot_from_df needs holoviews and bokeh, "
            "which are optional dependencies of nichecompass. Install them "
            "with ´pip install holoviews bokeh´.") from error

    hv.extension("bokeh")
    hv.output(size=200)

    sorted_cell_types = sorted(adata.obs[cell_type_key].unique().tolist())

    # Get group labels
    if (group_key is not None) & (groups == "all"):
        group_labels = df.index.get_level_values(
            df.index.names.index(group_key)).unique().tolist()
    elif (group_key is not None) & (groups != "all"):
        group_labels = groups
    else:
        group_labels = [""]

    chord_list = []
    for group_label in group_labels:
        if group_label == "":
            group_df = df
        else:
            group_df = df[df.index.get_level_values(
                df.index.names.index(group_key)) == group_label]
        
        # Get max value (over rows and columns) of the group for thresholding
        group_max = group_df.max().max()

        # Create group chord links
        links_list = []
        for i in range(len(sorted_cell_types)):
            for j in range(len(sorted_cell_types)):
                if group_df.iloc[i, j] > group_max * link_threshold:
                    link_dict = {}
                    link_dict["source"] = j
                    link_dict["target"] = i
                    link_dict["value"] = group_df.iloc[i, j]
                    links_list.append(link_dict)
        links = pd.DataFrame(links_list)

        # Create group chord nodes (only where links exist)
        nodes_list = []
        nodes_idx = []
        for i, cell_type in enumerate(sorted_cell_types):
            if i in (links["source"].values) or i in (links["target"].values):
                nodes_idx.append(i)
                nodes_dict = {}
                nodes_dict["name"] = cell_type
                nodes_dict["group"] = 1
                nodes_list.append(nodes_dict)
        nodes = hv.Dataset(pd.DataFrame(nodes_list, index=nodes_idx), "index")

        # Create group chord plot
        chord = hv.Chord((links, nodes)).select(value=(5, None))
        chord.opts(hv.opts.Chord(cmap="Category20",
                                 edge_cmap="Category20",
                                 edge_color=hv.dim("source").str(),
                                 labels="name",
                                 node_color=hv.dim("index").str(),
                                 title=f"{plot_label} {group_label}"))
        chord_list.append(chord)
    
    # Display chord plots
    layout = hv.Layout(chord_list).cols(2)
    hv.output(layout)

    # Save chord plots
    if save_fig:
        hv.save(layout,
                file_path,
                fmt="png")

        
def generate_enriched_gp_info_plots(plot_label: str,
                                    model: NicheCompass,
                                    sample_key: str,
                                    differential_gp_test_results_key: str,
                                    cat_key: str,
                                    cat_palette: dict,
                                    n_top_enriched_gp_start_idx: int=0,
                                    n_top_enriched_gp_end_idx: int=10,
                                    feature_spaces: list=["latent"],
                                    n_top_genes_per_gp: int=3,
                                    n_top_peaks_per_gp: int=0,
                                    scale_omics_ft: bool=False,
                                    save_figs: bool=False,
                                    figure_folder_path: str="",
                                    file_format: str="png",
                                    spot_size: float=30.,
                                    orientation=None):
    """
    Generate info plots of enriched gene programs. These show the enriched
    category, the gp activities, as well as the counts (or log normalized
    counts) of the top genes and/or peaks in a specified feature space.
    
    Parameters
    ----------
    plot_label:
        Main label of the plots.
    model:
        A trained NicheCompass model.
    sample_key:
        Key in ´adata.obs´ where the samples are stored.
    differential_gp_test_results_key:
        Key in ´adata.uns´ where the results of the differential gene program
        testing are stored.
    cat_key:
        Key in ´adata.obs´ where the categories that are used as colors for the
        enriched category plot are stored.
    cat_palette:
        Dictionary of colors that are used to highlight the categories, where
        the category is the key of the dictionary and the color is the value.
    n_top_enriched_gp_start_idx:
        Number of top enriched gene program from which to start the creation
        of plots.
    n_top_enriched_gp_end_idx:
        Number of top enriched gene program at which to stop the creation
        of plots.
    feature_spaces:
        List of feature spaces used for the info plots. Can be ´latent´ to use
        the latent embeddings for the plots, or it can be any of the samples
        stored in ´adata.obs[sample_key]´ to use the respective physical
        feature space for the plots.
    n_top_genes_per_gp:
        Number of top genes per gp to be considered in the info plots.
    n_top_peaks_per_gp:
        Number of top peaks per gp to be considered in the info plots. If ´>0´,
        requires the model to be trained inlcuding ATAC modality.
    scale_omics_ft:
        If ´True´, scale genes and peaks before plotting.
    save_figs:
        If ´True´, save the figures.
    figure_folder_path:
        Folder path where the figures will be saved.
    file_format:
        Format with which the figures will be saved.
    spot_size:
        Spot size used for the spatial plots.
    """
    model._check_if_trained(warn=True)
    orientation = model._gp_orientation(orientation)
    params = model.adata.uns.get(differential_gp_test_results_key + "_params", {})
    expected_id = "raw"
    if orientation == "canonical":
        model._gp_analysis_table()
        expected_id = model.gp_analysis_["orientation_id"]
    if params.get("orientation_id", "raw") != expected_id:
        raise ValueError("Differential results use a different or stale GP orientation. "
                         "Rerun run_differential_gp_tests before plotting.")
    if "input_fingerprint" in params:
        if (params.get("cat_key") != cat_key or
                params["input_fingerprint"] != model._gp_input_fingerprint(model.adata, cat_key)):
            raise ValueError("Differential results are stale after model, data, or group changes. "
                             "Rerun run_differential_gp_tests before plotting.")

    adata = model.adata.copy()
    if n_top_peaks_per_gp > 0:
        if "atac" not in model.modalities_:
            raise ValueError("The model needs to be trained with ATAC data if"
                             "'n_top_peaks_per_gp' > 0.")
        adata_atac = model.adata_atac.copy()
    
    # TODO
    if scale_omics_ft:
        sc.pp.scale(adata)
        if n_top_peaks_per_gp > 0:
            sc.pp.scale(adata_atac)
        adata.uns["omics_ft_pos_cmap"] = "RdBu"
        adata.uns["omics_ft_neg_cmap"] = "RdBu_r"
    else:
        if n_top_peaks_per_gp > 0 and sp.issparse(adata_atac.X):
            adata_atac.X = adata_atac.X.toarray()
        adata.uns["omics_ft_pos_cmap"] = "Blues"
        adata.uns["omics_ft_neg_cmap"] = "Reds"
        
    cats = list(adata.uns[differential_gp_test_results_key]["category"][
        n_top_enriched_gp_start_idx:n_top_enriched_gp_end_idx])
    gps = list(adata.uns[differential_gp_test_results_key]["gene_program"][
        n_top_enriched_gp_start_idx:n_top_enriched_gp_end_idx])
    if not gps:
        return
    log_bayes_factors = list(adata.uns[differential_gp_test_results_key]["log_bayes_factor"][
        n_top_enriched_gp_start_idx:n_top_enriched_gp_end_idx])
    
    if gps:
        unique_gps = list(dict.fromkeys(gps))
        adata.obs[unique_gps] = model.get_gp_activities(unique_gps, orientation=orientation)

    for gp in gps:
        # Get source and target genes, gene importances and gene signs and store
        # in temporary adata
        gp_gene_importances_df = model.compute_gp_gene_importances(
            selected_gp=gp, orientation=orientation)
        
        gp_source_genes_gene_importances_df = gp_gene_importances_df[
            gp_gene_importances_df["gene_entity"] == "source"]
        gp_target_genes_gene_importances_df = gp_gene_importances_df[
            gp_gene_importances_df["gene_entity"] == "target"]
        adata.uns["n_top_source_genes"] = n_top_genes_per_gp
        adata.uns[f"{gp}_source_genes_top_genes"] = (
            gp_source_genes_gene_importances_df["gene"][
                :n_top_genes_per_gp].values)
        adata.uns[f"{gp}_source_genes_top_gene_importances"] = (
            gp_source_genes_gene_importances_df["gene_importance"][
                :n_top_genes_per_gp].values)
        adata.uns[f"{gp}_source_genes_top_gene_signs"] = (
            np.where(gp_source_genes_gene_importances_df[
                "gene_weight"] > 0, "+", "-"))
        adata.uns["n_top_target_genes"] = n_top_genes_per_gp
        adata.uns[f"{gp}_target_genes_top_genes"] = (
            gp_target_genes_gene_importances_df["gene"][
                :n_top_genes_per_gp].values)
        adata.uns[f"{gp}_target_genes_top_gene_importances"] = (
            gp_target_genes_gene_importances_df["gene_importance"][
                :n_top_genes_per_gp].values)
        adata.uns[f"{gp}_target_genes_top_gene_signs"] = (
            np.where(gp_target_genes_gene_importances_df[
                "gene_weight"] > 0, "+", "-"))

        if n_top_peaks_per_gp > 0:
            # Get source and target peaks, peak importances and peak signs and
            # store in temporary adata
            gp_peak_importances_df = model.compute_gp_peak_importances(
                selected_gp=gp, orientation=orientation)
            gp_source_peaks_peak_importances_df = gp_peak_importances_df[
                gp_peak_importances_df["peak_entity"] == "source"]
            gp_target_peaks_peak_importances_df = gp_peak_importances_df[
                gp_peak_importances_df["peak_entity"] == "target"]
            adata.uns["n_top_source_peaks"] = n_top_peaks_per_gp
            adata.uns[f"{gp}_source_peaks_top_peaks"] = (
                gp_source_peaks_peak_importances_df["peak"][
                    :n_top_peaks_per_gp].values)
            adata.uns[f"{gp}_source_peaks_top_peak_importances"] = (
                gp_source_peaks_peak_importances_df["peak_importance"][
                    :n_top_peaks_per_gp].values)
            adata.uns[f"{gp}_source_peaks_top_peak_signs"] = (
                np.where(gp_source_peaks_peak_importances_df[
                    "peak_weight"] > 0, "+", "-"))
            adata.uns["n_top_target_peaks"] = n_top_peaks_per_gp
            adata.uns[f"{gp}_target_peaks_top_peaks"] = (
                gp_target_peaks_peak_importances_df["peak"][
                    :n_top_peaks_per_gp].values)
            adata.uns[f"{gp}_target_peaks_top_peak_importances"] = (
                gp_target_peaks_peak_importances_df["peak_importance"][
                    :n_top_peaks_per_gp].values)
            adata.uns[f"{gp}_target_peaks_top_peak_signs"] = (
                np.where(gp_target_peaks_peak_importances_df[
                    "peak_weight"] > 0, "+", "-"))
            
            # Add peak counts to temporary adata for plotting
            adata.obs[[peak for peak in 
                       adata.uns[f"{gp}_target_peaks_top_peaks"]]] = (
                adata_atac.X[
                    :, [adata_atac.var_names.tolist().index(peak)
                        for peak in adata.uns[f"{gp}_target_peaks_top_peaks"]]])
            adata.obs[[peak for peak in
                       adata.uns[f"{gp}_source_peaks_top_peaks"]]] = (
                adata_atac.X[
                    :, [adata_atac.var_names.tolist().index(peak)
                        for peak in adata.uns[f"{gp}_source_peaks_top_peaks"]]])
        else:
            adata.uns["n_top_source_peaks"] = 0
            adata.uns["n_top_target_peaks"] = 0

    for feature_space in feature_spaces:
        plot_enriched_gp_info_plots_(
            adata=adata,
            sample_key=sample_key,
            gps=gps,
            log_bayes_factors=log_bayes_factors,
            cat_key=cat_key,
            cat_palette=cat_palette,
            cats=cats,
            feature_space=feature_space,
            spot_size=spot_size,
            suptitle=f"{plot_label.replace('_', ' ').title()} "
                     f"Top {n_top_enriched_gp_start_idx} to "
                     f"{n_top_enriched_gp_end_idx} Enriched GPs: "
                     f"GP Scores and Omics Feature Counts in "
                     f"{feature_space} Feature Space",
            save_fig=save_figs,
            figure_folder_path=figure_folder_path,
            fig_name=f"{plot_label}_top_{n_top_enriched_gp_start_idx}"
                     f"-{n_top_enriched_gp_end_idx}_enriched_gps_gp_scores_"
                     f"omics_feature_counts_in_{feature_space}_"
                     f"feature_space.{file_format}")
            
            
def plot_enriched_gp_info_plots_(adata: AnnData,
                                 sample_key: str,
                                 gps: list,
                                 log_bayes_factors: list,
                                 cat_key: str,
                                 cat_palette: dict,
                                 cats: list,
                                 feature_space: str,
                                 spot_size: float,
                                 suptitle: str,
                                 save_fig: bool,
                                 figure_folder_path: str,
                                 fig_name: str):
    """
    This is a helper function to plot gene program info plots in a specified
    feature space.
    
    Parameters
    ----------
    adata:
        An AnnData object with stored information about the gene programs to be
        plotted.
    sample_key:
        Key in ´adata.obs´ where the samples are stored.
    gps:
        List of gene programs for which info plots will be created.
    log_bayes_factors:
        List of log bayes factors corresponding to gene programs
    cat_key:
        Key in ´adata.obs´ where the categories that are used as colors for the
        enriched category plot are stored.
    cat_palette:
        Dictionary of colors that are used to highlight the categories, where
        the category is the key of the dictionary and the color is the value.
    cats:
        List of categories for which the corresponding gene programs in ´gps´
        are enriched.
    feature_space:
        Feature space used for the plots. Can be ´latent´ to use the latent
        embeddings for the plots, or it can be any of the samples stored in
        ´adata.obs[sample_key]´ to use the respective physical feature space for
        the plots.
    spot_size:
        Spot size used for the spatial plots.
    subtitle:
        Overall figure title.
    save_fig:
        If ´True´, save the figure.
    figure_folder_path:
        Path of the folder where the figure will be saved.
    fig_name:
        Name of the figure under which it will be saved.
    """
    # Define figure configurations
    ncols = (2 +
             adata.uns["n_top_source_genes"] +
             adata.uns["n_top_target_genes"] +
             adata.uns["n_top_source_peaks"] +
             adata.uns["n_top_target_peaks"])
    fig_width = (12 + (6 * (
        adata.uns["n_top_source_genes"] +
        adata.uns["n_top_target_genes"] +
        adata.uns["n_top_source_peaks"] +
        adata.uns["n_top_target_peaks"])))
    wspace = 0.3
    fig, axs = plt.subplots(nrows=len(gps),
                            ncols=ncols,
                            figsize=(fig_width, 6*len(gps)))
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)
    title = fig.suptitle(t=suptitle,
                         x=0.55,
                         y=(1.1 if len(gps) == 1 else 0.97),
                         fontsize=20)
    
    # Plot enriched gp category and gene program latent scores
    for i, gp in enumerate(gps):
        parts = gp.split("_")
        spatial_gp_label = gp.replace("_", "\n", 1)
        latent_gp_label = (f"{parts[0]}\n{' '.join(parts[1:-1])}\n{parts[-1]}"
                           if len(parts) > 2 else spatial_gp_label)
        if feature_space == "latent":
            sc.pl.umap(
                adata,
                color=cat_key,
                palette=cat_palette,
                groups=cats[i],
                ax=axs[i, 0],
                title="Enriched GP Category",
                legend_loc="on data",
                na_in_legend=False,
                show=False)
            sc.pl.umap(
                adata,
                color=gps[i],
                color_map="RdBu",
                ax=axs[i, 1],
                title=f"{latent_gp_label} score (LBF: {round(log_bayes_factors[i])})",
                colorbar_loc="bottom",
                show=False)
        else:
            sc.pl.spatial(
                adata=adata[adata.obs[sample_key] == feature_space],
                color=cat_key,
                palette=cat_palette,
                groups=cats[i],
                ax=axs[i, 0],
                spot_size=spot_size,
                title="Enriched GP Category",
                legend_loc="on data",
                na_in_legend=False,
                show=False)
            sc.pl.spatial(
                adata=adata[adata.obs[sample_key] == feature_space],
                color=gps[i],
                color_map="RdBu",
                spot_size=spot_size,
                title=f"{spatial_gp_label} "
                      f"(LBF: {round(log_bayes_factors[i], 2)})",
                legend_loc=None,
                ax=axs[i, 1],
                colorbar_loc="bottom",
                show=False) 
        axs[i, 0].xaxis.label.set_visible(False)
        axs[i, 0].yaxis.label.set_visible(False)
        axs[i, 1].xaxis.label.set_visible(False)
        axs[i, 1].yaxis.label.set_visible(False)

        # Plot omics feature counts (or log normalized counts)
        modality_entities = []
        if len(adata.uns[f"{gp}_source_genes_top_genes"]) > 0:
            modality_entities.append("source_genes")
        if len(adata.uns[f"{gp}_target_genes_top_genes"]) > 0:
            modality_entities.append("target_genes")
        if f"{gp}_source_peaks_top_peaks" in adata.uns.keys():
            gp_n_source_peaks_top_peaks = (
                len(adata.uns[f"{gp}_source_peaks_top_peaks"]))
            if len(adata.uns[f"{gp}_source_peaks_top_peaks"]) > 0:
                modality_entities.append("source_peaks")
        else:
            gp_n_source_peaks_top_peaks = 0
        if f"{gp}_target_peaks_top_peaks" in adata.uns.keys():
            gp_n_target_peaks_top_peaks = (
                len(adata.uns[f"{gp}_target_peaks_top_peaks"]))
            if len(adata.uns[f"{gp}_target_peaks_top_peaks"]) > 0:
                modality_entities.append("target_peaks")
        else:
            gp_n_target_peaks_top_peaks = 0
        for modality_entity in modality_entities:
            # Define k for index iteration
            if modality_entity == "source_genes":
                k = 0
            elif modality_entity == "target_genes":
                k = len(adata.uns[f"{gp}_source_genes_top_genes"])
            elif modality_entity == "source_peaks":
                k = (len(adata.uns[f"{gp}_source_genes_top_genes"]) +
                     len(adata.uns[f"{gp}_target_genes_top_genes"]))
            elif modality_entity == "target_peaks":
                k = (len(adata.uns[f"{gp}_source_genes_top_genes"]) +
                     len(adata.uns[f"{gp}_target_genes_top_genes"]) +
                     len(adata.uns[f"{gp}_source_peaks_top_peaks"]))
            for j in range(len(adata.uns[f"{gp}_{modality_entity}_top_"
                                         f"{modality_entity.split('_')[1]}"])):
                if feature_space == "latent":
                    sc.pl.umap(
                        adata,
                        color=adata.uns[f"{gp}_{modality_entity}_top_"
                                        f"{modality_entity.split('_')[1]}"][j],
                        color_map=(adata.uns["omics_ft_pos_cmap"] if
                                   adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1][:-1]}"
                                             "_signs"][j] == "+" else adata.uns["omics_ft_neg_cmap"]),
                        ax=axs[i, 2+k+j],
                        legend_loc="on data",
                        na_in_legend=False,
                        title=f"""{adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1]}"
                                             ][j]}: """
                              f"""{adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1][:-1]}"
                                             "_importances"][j]:.2f} """
                              f"({modality_entity[:-1]}; "
                              f"""{adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1][:-1]}"
                                             "_signs"][j]})""",
                        colorbar_loc="bottom",
                        show=False)
                else:
                    sc.pl.spatial(
                        adata=adata[adata.obs[sample_key] == feature_space],
                        color=adata.uns[f"{gp}_{modality_entity}_top_"
                                        f"{modality_entity.split('_')[1]}"][j],
                        color_map=(adata.uns["omics_ft_pos_cmap"] if
                                   adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1][:-1]}"
                                             "_signs"][j] == "+" else adata.uns["omics_ft_neg_cmap"]),
                        legend_loc="on data",
                        na_in_legend=False,
                        ax=axs[i, 2+k+j],
                        spot_size=spot_size,
                        title=f"""{adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1]}"
                                             ][j]} \n"""
                              f"""({adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1][:-1]}"
                                             "_importances"][j]:.2f}; """
                              f"{modality_entity[:-1]}; "
                              f"""{adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1][:-1]}"
                                             "_signs"][j]})""",
                        colorbar_loc="bottom",
                        show=False)
                axs[i, 2+k+j].xaxis.label.set_visible(False)
                axs[i, 2+k+j].yaxis.label.set_visible(False)
            # Remove unnecessary axes
            for l in range(2 +
                           len(adata.uns[f"{gp}_source_genes_top_genes"]) +
                           len(adata.uns[f"{gp}_target_genes_top_genes"]) +
                           gp_n_source_peaks_top_peaks +
                           gp_n_target_peaks_top_peaks, ncols):
                axs[i, l].set_visible(False)

    # Save and display plot
    plt.subplots_adjust(wspace=wspace, hspace=0.275)
    if save_fig:
        fig.savefig(f"{figure_folder_path}/{fig_name}",
                    bbox_extra_artists=(title,),
                    bbox_inches="tight")
    plt.show()

default_color_dict = {
    "0": "#66C5CC",
    "1": "#F6CF71",
    "2": "#F89C74",
    "3": "#DCB0F2",
    "4": "#87C55F",
    "5": "#9EB9F3",
    "6": "#FE88B1",
    "7": "#C9DB74",
    "8": "#8BE0A4",
    "9": "#B497E7",
    "10": "#D3B484",
    "11": "#B3B3B3",
    "12": "#276A8C", # Royal Blue
    "13": "#DAB6C4", # Pink
    "14": "#C38D9E", # Mauve-Pink
    "15": "#9D88A2", # Mauve
    "16": "#FF4D4D", # Light Red
    "17": "#9B4DCA", # Lavender-Purple
    "18": "#FF9CDA", # Bright Pink
    "19": "#FF69B4", # Hot Pink
    "20": "#FF00FF", # Magenta
    "21": "#DA70D6", # Orchid
    "22": "#BA55D3", # Medium Orchid
    "23": "#8A2BE2", # Blue Violet
    "24": "#9370DB", # Medium Purple
    "25": "#7B68EE", # Medium Slate Blue
    "26": "#4169E1", # Royal Blue
    "27": "#FF8C8C", # Salmon Pink
    "28": "#FFAA80", # Light Coral
    "29": "#48D1CC", # Medium Turquoise
    "30": "#40E0D0", # Turquoise
    "31": "#00FF00", # Lime
    "32": "#7FFF00", # Chartreuse
    "33": "#ADFF2F", # Green Yellow
    "34": "#32CD32", # Lime Green
    "35": "#228B22", # Forest Green
    "36": "#FFD8B8", # Peach
    "37": "#008080", # Teal
    "38": "#20B2AA", # Light Sea Green
    "39": "#00FFFF", # Cyan
    "40": "#00BFFF", # Deep Sky Blue
    "41": "#3D5A80", # Royal Blue
    "42": "#0000CD", # Medium Blue
    "43": "#00008B", # Dark Blue
    "44": "#8B008B", # Dark Magenta
    "45": "#FF1493", # Deep Pink
    "46": "#FF4500", # Orange Red
    "47": "#006400", # Dark Green
    "48": "#FF6347", # Tomato
    "49": "#FF7F50", # Coral
    "50": "#CD5C5C", # Indian Red
    "51": "#B22222", # Fire Brick
    "52": "#FFB83F",  # Light Orange
    "53": "#8B0000", # Dark Red
    "54": "#D2691E", # Chocolate
    "55": "#A0522D", # Sienna
    "56": "#800000", # Maroon
    "57": "#808080", # Gray
    "58": "#A9A9A9", # Dark Gray
    "59": "#C0C0C0", # Silver
    "60": "#9DD84A",
    "61": "#F5F5F5", # White Smoke
    "62": "#F17171", # Light Red
    "63": "#000000", # Black
    "64": "#FF8C42", # Tangerine
    "65": "#F9A11F", # Bright Orange-Yellow
    "66": "#FACC15", # Golden Yellow
    "67": "#E2E062", # Pale Lime
    "68": "#BADE92", # Soft Lime
    "69": "#70C1B3", # Greenish-Blue
    "70": "#41B3A3", # Turquoise
    "71": "#5EAAA8", # Gray-Green
    "72": "#72B01D", # Chartreuse
    "73": "#9CD08F", # Light Green
    "74": "#8EBA43", # Olive Green
    "75": "#FAC8C3", # Light Pink
    "76": "#E27D60", # Dark Salmon
    "77": "#9E6B7F", # Mauve-Pink
    "78": "#937D64", # Light Brown
    "79": "#B1C1CC", # Light Blue-Gray
    "80": "#88A0A8", # Gray-Blue-Green
    "81": "#4E598C", # Dark Blue-Purple
    "82": "#4B4E6D", # Dark Gray-Blue
    "83": "#8E9AAF", # Light Blue-Grey
    "84": "#C0D6DF", # Pale Blue-Grey
    "85": "#97C1A9", # Blue-Green
    "86": "#4C6E5D", # Dark Green
    "87": "#95B9C7", # Pale Blue-Green
    "88": "#C1D5E0", # Pale Gray-Blue
    "89": "#ECDB54", # Bright Yellow
    "90": "#E89B3B", # Bright Orange
    "91": "#CE5A57", # Deep Red
    "92": "#C3525A", # Dark Red
    "93": "#B85D8E", # Berry
    "94": "#7D5295", # Deep Purple
    "-1" : "#E1D9D1",
    "None" : "#E1D9D1"
}

_CATEGORY_PALETTES = {
    "cell_type_28":
            ["#023fa5",
             "#7d87b9",
             "#bec1d4",
             "#d6bcc0",
             "#bb7784",
             "#8e063b",
             "#4a6fe3",
             "#8595e1",
             "#b5bbe3",
             "#e6afb9",
             "#e07b91",
             "#d33f6a",
             "#11c638",
             "#8dd593",
             "#c6dec7",
             "#ead3c6",
             "#f0b98d",
             "#ef9708",
             "#0fcfc0",
             "#9cded6",
             "#d5eae7",
             "#f3e1eb",
             "#f6c4e1",
             "#f79cd4",
             '#7f7f7f',
             "#c7c7c7",
             "#1CE6FF",
             "#336600"],
    "cell_type_20":
            ['#1f77b4',
             '#ff7f0e',
             '#279e68',
             '#d62728',
             '#aa40fc',
             '#8c564b',
             '#e377c2',
             '#b5bd61',
             '#17becf',
             '#aec7e8',
             '#ffbb78',
             '#98df8a',
             '#ff9896',
             '#c5b0d5',
             '#c49c94',
             '#f7b6d2',
             '#dbdb8d',
             '#9edae5',
             '#ad494a',
             '#8c6d31'],
    "cell_type_10":
            ['#7f7f7f',
             '#ff7f0e',
             '#279e68',
             '#e377c2',
             '#17becf',
             '#8c564b',
             '#d62728',
             '#1f77b4',
             '#b5bd61',
             '#aa40fc'],
    "batch":
            ['#0173b2', '#d55e00', '#ece133', '#ca9161', '#fbafe4',
             '#949494', '#de8f05', '#029e73', '#cc78bc', '#56b4e9',
             '#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF',
             '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF',
             '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
             '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C'],
}

# The 28 color palette was historically named "cell_type_30"; both
# names resolve to it, and the exhaustion warning now reports the real
# count rather than the one the name implies.
_CATEGORY_PALETTES["cell_type_30"] = _CATEGORY_PALETTES["cell_type_28"]


def create_new_color_dict(
        adata,
        cat_key,
        color_palette="default",
        overwrite_color_dict={"-1" : "#E1D9D1"},
        skip_default_colors=0):
    """
    Create a dictionary of color hexcodes for a specified category.

    Parameters
    ----------
    adata:
        AnnData object.
    cat_key:
        Key in ´adata.obs´ where the categories are stored for which color
        hexcodes will be created.
    color_palette:
        Type of color palette.
    overwrite_color_dict:
        Dictionary with overwrite values that will take precedence over the
        automatically created dictionary.
    skip_default_colors:
        Number of colors to skip from the default color dict.

    Returns
    ----------
    new_color_dict:
        The color dictionary with a hexcode for each category.
    """
    # Order of appearance, deliberately left alone. Sorting would change
    # which color every category gets, churning every existing figure,
    # and it buys nothing: the returned mapping is consumed by NAME
    # (scanpy takes it as ´palette=´ and looks categories up), so the
    # ordering carries no correctness. It does NOT make colors stable
    # across subsets of a dataset either - any positional assignment
    # shifts when a category is absent. Pass ´overwrite_color_dict´ to
    # pin specific categories.
    new_categories = adata.obs[cat_key].unique().tolist()
    if color_palette == "default":
        palette = list(default_color_dict.values())[skip_default_colors:]
    elif color_palette in _CATEGORY_PALETTES:
        palette = _CATEGORY_PALETTES[color_palette]
    else:
        raise ValueError(
            f"´color_palette´ is {color_palette!r}, which is not one of "
            f"{sorted(('default', *_CATEGORY_PALETTES))}.")

    # ´zip´ would truncate here, silently leaving later categories out of the
    # returned dict. Downstream that surfaces either as a ´KeyError´ or as
    # scanpy substituting its own colors, so two figures of the same data
    # disagree about which category is which color.
    if len(new_categories) > len(palette):
        warnings.warn(
            f"´{cat_key}´ has {len(new_categories)} categories but the "
            f"{color_palette!r} palette has {len(palette)} colors, so colors "
            "are reused and some categories are indistinguishable. Pass a "
            "larger palette, or override the repeats via "
            "´overwrite_color_dict´.")
    new_color_dict = {category: palette[i % len(palette)]
                      for i, category in enumerate(new_categories)}
    for key, val in overwrite_color_dict.items():
        new_color_dict[key] = val
    return new_color_dict


def plot_non_zero_gene_count_means_dist(
        adata: AnnData,
        genes: list,
        gene_label: str):
    """
    Plot distribution of non zero gene count means in the adata over all 
    specified genes.
    """
    gene_counts = adata[
        :, [gene for gene in adata.var_names if gene in genes]].layers["counts"]
    nz_gene_means = np.mean(
        np.ma.masked_equal(gene_counts.toarray(), 0), axis=0).data
    
    sns.kdeplot(nz_gene_means)
    plt.title(f"{gene_label} Genes Average Non-Zero Gene Counts per Gene")
    plt.xlabel("Average Non-zero Gene Counts")
    plt.ylabel("Gene Density")
    plt.show()


def _communication_spatial_graph(adata, n_neighbors, sample_key):
    """Build a spatial graph within samples and validate its cached inputs."""
    import hashlib

    if (isinstance(n_neighbors, (bool, np.bool_))
            or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 2):
        raise ValueError("n_neighbors must be an integer of at least 2.")
    if sample_key is not None:
        if sample_key not in adata.obs:
            raise ValueError(f"Unknown sample_key: {sample_key!r}.")
        labels = adata.obs[sample_key]
        if labels.isna().any():
            raise ValueError("Communication sample labels must not be missing.")
        sample_codes, _ = pd.factorize(labels, sort=False)
    else:
        labels = pd.Series(np.zeros(adata.n_obs, dtype=int), index=adata.obs_names)
        sample_codes = np.zeros(adata.n_obs, dtype=int)

    cached = adata.uns.get("spatial_cci", {})
    params = cached.get("params", {})
    graph_key = "spatial_cci_connectivities"
    valid_cached_shape = (graph_key in adata.obsp
                          and adata.obsp[graph_key].shape == (adata.n_obs, adata.n_obs))
    if "spatial" not in adata.obsm:
        # Historical callers may supply a graph directly without coordinates.
        # Only allow that convention when no sample partition was requested.
        if (sample_key is None and valid_cached_shape
                and params.get("n_neighbors") == n_neighbors
                and "input_fingerprint" not in cached):
            return adata.obsp[graph_key]
        raise ValueError("Communication requires spatial coordinates in adata.obsm['spatial'].")
    coordinates = np.asarray(adata.obsm["spatial"])
    if (coordinates.ndim != 2 or coordinates.shape[1] == 0
            or not np.issubdtype(coordinates.dtype, np.number)
            or not np.isfinite(coordinates).all()):
        raise ValueError("Communication spatial coordinates must be a finite numeric matrix.")
    digest = hashlib.sha256()
    digest.update(repr((coordinates.shape, str(coordinates.dtype), n_neighbors, sample_key)).encode())
    digest.update(np.ascontiguousarray(coordinates).tobytes())
    digest.update(pd.util.hash_pandas_object(labels, index=True).values.tobytes())
    fingerprint = digest.hexdigest()
    if valid_cached_shape and cached.get("input_fingerprint") == fingerprint:
        return adata.obsp[graph_key]

    blocks = {"connectivities": [], "distances": []}
    for code in pd.unique(sample_codes):
        indices = np.flatnonzero(sample_codes == code)
        if len(indices) < 2:
            continue
        if len(indices) == 2:
            rows, cols = np.array([0, 1]), np.array([1, 0])
            distance = np.linalg.norm(coordinates[indices[0]] - coordinates[indices[1]])
            local = {"connectivities": sp.coo_matrix((np.ones(2), (rows, cols)), shape=(2, 2)),
                     "distances": sp.coo_matrix((np.full(2, distance), (rows, cols)), shape=(2, 2))}
        else:
            subset = AnnData(np.zeros((len(indices), 0), dtype=np.float32))
            subset.obsm["spatial"] = coordinates[indices]
            sc.pp.neighbors(subset, n_neighbors=min(n_neighbors, len(indices)),
                            use_rep="spatial", key_added="spatial_cci")
            local = {kind: subset.obsp[f"spatial_cci_{kind}"].tocoo() for kind in blocks}
        for kind, block in local.items():
            blocks[kind].append((indices[block.row], indices[block.col], block.data))
    for kind, pieces in blocks.items():
        if pieces:
            row, col, data = (np.concatenate([piece[i] for piece in pieces]) for i in range(3))
            graph = sp.csr_matrix((data, (row, col)), shape=(adata.n_obs, adata.n_obs))
        else:
            graph = sp.csr_matrix((adata.n_obs, adata.n_obs), dtype=np.float32)
        adata.obsp[f"spatial_cci_{kind}"] = graph
    adata.uns["spatial_cci"] = {
        "connectivities_key": graph_key, "distances_key": "spatial_cci_distances",
        "params": {"n_neighbors": n_neighbors, "use_rep": "spatial", "method": "umap"},
        "sample_key": sample_key or "", "input_fingerprint": fingerprint}
    return adata.obsp[graph_key]


def compute_communication_gp_network(
    gp_list: list,
    model: NicheCompass,
    group_key: str="niche",
    filter_key: Optional[str]=None,
    filter_cat: Optional[str]=None,
    n_neighbors: int=90,
    sample_key: Optional[str]=None,
    normalize: Literal["global", "per_gp", "none"]="global",
    store_scores: bool=True):
    """
    Compute a network of category aggregated cell-pair communication strengths.

    For every gene program, each cell gets a source score and a target score:
    the expression of the program's source (respectively target) genes,
    normalized per gene by its maximum across cells, weighted by that gene's
    decoder weight, averaged over genes, and scaled by the cell's gene program
    activity. Negative averages are clipped to zero. The two scores are then
    multiplied along the edges of a spatial neighbor graph and aggregated per
    group.

    Parameters
    ----------
    gp_list:
        List of GPs for which the cell-pair communication strengths are
        computed.
    model:
        A trained NicheCompass model.
    group_key:
        Key in ´adata.obs´ where the groups are stored over which the cell-pair
        communication strengths will be aggregated.
    filter_key:
        Key in ´adata.obs´ that contains the category for which the results are
        filtered. The filter applies to the SENDING cell.
    filter_cat:
        Category for which the results are filtered.
    n_neighbors:
        Number of neighbors for the gp-specific neighborhood graph.
    sample_key:
        Observation column identifying independent spatial samples. Neighbors
        are computed within each sample. Pass this for integrated datasets;
        otherwise all observations are treated as one spatial sample.
    normalize:
        How ´strength´ is scaled. ´"global"´ (default) divides by the largest
        aggregated value across ALL selected gene programs, which keeps widths
        comparable when the programs are drawn on one axes. ´"per_gp"´ divides
        by the largest value within each program, which shows within-program
        rank only. ´"none"´ leaves the aggregated values untouched. In every
        case zero means no communication: unlike a min-max rescaling, nothing
        is subtracted, so the weakest pair is not forced to zero and dropped.
    store_scores:
        If ´True´ (default), write the per-cell scores to
        ´adata.obs["<gp>_source_score"]´ and ´adata.obs["<gp>_target_score"]´
        and the edge products to ´adata.obsp["<gp>_connectivities"]´. Set to
        ´False´ to leave ´adata´ untouched, which matters when many gene
        programs are requested, since each adds an n_obs x n_obs sparse
        matrix.

    Returns
    ----------
    network_df:
        A pandas dataframe with one row per (source group, target group, gene
        program) and columns ´source´, ´target´, ´strength´,
        ´strength_unscaled´ and ´edge_type´. Pairs with no communication are
        omitted.
    """
    # Validate before creating graphs or writing communication scores. These
    # scores require a prior with both measured source and target members.
    gp_list, gp_indices = model._gp_selection(gp_list)
    if not gp_list:
        raise ValueError("Select at least one communication GP.")
    if normalize not in ("global", "per_gp", "none"):
        raise ValueError(
            f"´normalize´ is {normalize!r}, which is not one of 'global', "
            "'per_gp' or 'none'.")
    if filter_key is not None and filter_cat is None:
        raise ValueError("´filter_cat´ is required when ´filter_key´ is set.")
    if group_key not in model.adata.obs:
        raise ValueError(f"´group_key´ {group_key!r} is not a column of "
                         "´adata.obs´.")
    gp_summary_df = model.get_gp_summary(orientation="raw")
    selected_summary = gp_summary_df.set_index("gp_name").loc[gp_list]
    if (np.any(gp_indices >= model.n_prior_gp_)
            or not selected_summary.gp_active.all()
            or (selected_summary.n_source_genes == 0).any()
            or (selected_summary.n_target_genes == 0).any()):
        raise ValueError("Communication requires active prior GPs with source and target genes.")
    # Products require scores and weights in the same coordinate convention.
    raw_scores = model.get_gp_activities(gp_list, orientation="raw", use_cached=True)
    spatial_graph = _communication_spatial_graph(model.adata, n_neighbors, sample_key)

    # Hoisted out of the gene program loop: the edge list of the spatial graph
    # does not depend on the gene program.
    edges = (spatial_graph > 0).tocoo()
    edge_rows, edge_cols = edges.row, edges.col

    # Group codes, once. Aggregating straight into a (n_groups x n_groups)
    # matrix from the edge list avoids the (n_obs x n_groups) intermediate that
    # ´aggregate_obsp_matrix_per_cell_type´ builds per gene program.
    groups = model.adata.obs[group_key]
    group_cats = sorted(groups.unique().tolist())
    group_codes = groups.map(
        {cat: i for i, cat in enumerate(group_cats)}).to_numpy()
    n_groups = len(group_cats)
    if filter_key is not None:
        if filter_key not in model.adata.obs:
            raise ValueError(f"´filter_key´ {filter_key!r} is not a column of "
                             "´adata.obs´.")
        filter_values = model.adata.obs[filter_key].astype(str).to_numpy()
        if str(filter_cat) not in set(filter_values):
            raise ValueError(
                f"´filter_cat´ {filter_cat!r} does not appear in "
                f"´adata.obs[{filter_key!r}]´.")
        # The filter is a property of the sending cell, i.e. of the edge's row.
        edge_keep = filter_values[edge_rows] == str(filter_cat)
        edge_rows, edge_cols = edge_rows[edge_keep], edge_cols[edge_keep]
    else:
        edge_keep = None

    # One pass over the count matrix for the union of every gene involved,
    # instead of an AnnData view per gene. A single column slice of a CSR
    # matrix costs O(nnz) on its own, so the per-gene version was O(n_genes x
    # nnz) overall.
    targets_mask = model.adata.varm[model.gp_targets_categories_mask_key_]
    sources_mask = model.adata.varm[model.gp_sources_categories_mask_key_]
    gene_idx_per_gp = {}
    all_genes = set()
    for gp_column, gp in enumerate(gp_list):
        gp_idx = gp_indices[gp_column]
        source_genes_idx = np.flatnonzero(sources_mask[:, gp_idx])
        target_genes_idx = np.flatnonzero(targets_mask[:, gp_idx])
        gene_idx_per_gp[gp] = (source_genes_idx, target_genes_idx)
        all_genes.update(source_genes_idx.tolist())
        all_genes.update(target_genes_idx.tolist())
    gene_order = np.array(sorted(all_genes), dtype=np.int64)
    gene_position = {gene: i for i, gene in enumerate(gene_order)}
    counts_sub = model.adata.X[:, gene_order]
    if sp.issparse(counts_sub):
        counts_sub = counts_sub.tocsc()
        gene_max = np.asarray(counts_sub.max(axis=0).todense()).ravel()
    else:
        counts_sub = np.asarray(counts_sub)
        gene_max = counts_sub.max(axis=0)

    def _aggregated_score(genes_idx, gene_names, weights, gp_scores):
        """
        Per-cell score for one arm of a gene program.

        Each column of the matrix the original built was
        ´counts_i / max_i * w_i * gp_scores´, and only its mean over genes was
        ever read, so the mean is accumulated directly:
        ´gp_scores * (counts @ (w / max)) / n_genes´. That is exact, and it
        avoids allocating an (n_obs x n_genes) float64 scratch matrix per arm.
        """
        weight_of = dict(zip(gene_names, weights))
        columns = np.array([gene_position[gene] for gene in genes_idx],
                           dtype=np.int64)
        maxima = gene_max[columns]
        coefficients = np.array(
            [weight_of[model.adata.var_names[gene]] for gene in genes_idx],
            dtype=np.float64)
        # A gene that is zero in every cell contributed nothing before, and
        # dividing by its maximum would be a division by zero.
        coefficients = np.where(maxima > 0,
                                coefficients / np.where(maxima > 0, maxima, 1),
                                0.)
        weighted = counts_sub[:, columns] @ coefficients
        weighted = np.asarray(weighted).ravel()
        score = (gp_scores * weighted / len(genes_idx)).astype("float32")
        # Clipped after the mean over genes, as before: a negative average is
        # not a negative amount of communication.
        np.clip(score, 0., None, out=score)
        return score

    gp_network_dfs = []
    for gp_column, gp in enumerate(gp_list):
        gp_scores = raw_scores[:, gp_column]
        summary_row = selected_summary.loc[gp]
        source_genes_idx, target_genes_idx = gene_idx_per_gp[gp]

        agg_gp_source_score = _aggregated_score(
            source_genes_idx,
            summary_row["gp_source_genes"],
            summary_row["gp_source_genes_weights"],
            gp_scores)
        agg_gp_target_score = _aggregated_score(
            target_genes_idx,
            summary_row["gp_target_genes"],
            summary_row["gp_target_genes_weights"],
            gp_scores)

        products = agg_gp_source_score[edge_rows] * agg_gp_target_score[edge_cols]

        if store_scores:
            model.adata.obs[f"{gp}_source_score"] = agg_gp_source_score
            model.adata.obs[f"{gp}_target_score"] = agg_gp_target_score
            if edge_keep is None:
                connectivity_rows, connectivity_cols = edge_rows, edge_cols
                connectivity_values = products
            else:
                connectivity_rows = edges.row
                connectivity_cols = edges.col
                connectivity_values = (
                    agg_gp_source_score[connectivity_rows]
                    * agg_gp_target_score[connectivity_cols])
            connectivities = sp.csr_matrix(
                (connectivity_values, (connectivity_rows, connectivity_cols)),
                shape=spatial_graph.shape)
            # Building from a COO triplet keeps explicit zeros, so a gene
            # program with no communication would still store one entry per
            # spatial edge. The old code shed them by accident, inside
            # ´aggregate_obsp_matrix_per_cell_type´, which this no longer
            # calls.
            connectivities.eliminate_zeros()
            model.adata.obsp[f"{gp}_connectivities"] = connectivities

        # Rows are the sending group, columns the receiving group.
        matrix = np.zeros((n_groups, n_groups), dtype=np.float64)
        np.add.at(matrix,
                  (group_codes[edge_rows], group_codes[edge_cols]),
                  products)

        gp_network_df = pd.DataFrame(
            {"source": np.repeat(group_cats, n_groups),
             "target": np.tile(group_cats, n_groups),
             "strength_unscaled": matrix.ravel()})
        gp_network_df["edge_type"] = gp
        gp_network_dfs.append(gp_network_df)

    network_df = pd.concat(gp_network_dfs, ignore_index=True)
    # A pair with no communication is dropped; a pair with the weakest nonzero
    # communication is not. A min-max rescaling subtracts the minimum, which
    # maps the weakest pair to exactly zero and, with the subsequent positivity
    # filter, discarded it on every call however strong it really was.
    network_df = network_df[network_df["strength_unscaled"] > 0].copy()
    if normalize == "none":
        network_df["strength"] = network_df["strength_unscaled"]
    else:
        if normalize == "global":
            denominators = network_df["strength_unscaled"].max()
        else:
            denominators = network_df.groupby("edge_type")[
                "strength_unscaled"].transform("max")
        network_df["strength"] = np.where(
            denominators > 0,
            network_df["strength_unscaled"] / denominators,
            0.)
    network_df = network_df.sort_values(
        "strength_unscaled", ascending=False).reset_index(drop=True)
    return network_df[["source", "target", "strength", "strength_unscaled",
                       "edge_type"]]



def visualize_communication_gp_network(
    adata,
    network_df,
    cat_colors,
    edge_type_colors: Optional[dict]=None,
    edge_width_scale: float=20.0,
    node_size: int=500,
    fontsize: int=14,
    figsize: Tuple[int, int]=(18, 16),
    plot_legend: bool=True,
    save: bool=False,
    save_path: str="communication_gp_network.svg",
    show: bool=True,
    text_space: float=1.3,
    connection_style="arc3, rad = 0.1",
    cat_key: str="niche",
    edge_attr: str="strength",
    reserve_all_categories: bool=False,
    ax=None):
    """
    Visualize a communication gene program network as a circular digraph.

    Parameters
    ----------
    adata:
        AnnData object the network was computed from. Only used to order the
        nodes, and to reserve a slot for every category when
        ´reserve_all_categories´ is ´True´.
    network_df:
        Output of ´compute_communication_gp_network´, with columns ´source´,
        ´target´, ´edge_type´ and the column named by ´edge_attr´.
    cat_colors:
        Mapping from category to color. Must cover every node in the network.
    edge_type_colors:
        Mapping from gene program name to color, or a list of colors to assign
        in sorted gene program order. Defaults to a 20 color palette, cycled
        with a warning if there are more gene programs than colors.
    edge_width_scale:
        Multiplier applied to ´edge_attr´ to get the drawn line width.
    reserve_all_categories:
        If ´True´, every category of ´adata.obs[cat_key]´ gets a slot on the
        circle even when it has no edges, so that figures of different gene
        programs are directly superimposable. If ´False´ (default) only the
        categories present in the network are drawn, spaced evenly.
    ax:
        Axes to draw on. A new figure is created if this is ´None´.

    Returns
    ----------
    ax:
        The axes drawn on.
    """
    required = {"source", "target", "edge_type", edge_attr}
    missing_cols = required.difference(network_df.columns)
    if missing_cols:
        raise ValueError(
            f"´network_df´ is missing the columns {sorted(missing_cols)}.")
    if len(network_df) == 0:
        raise ValueError(
            "´network_df´ is empty, so there is nothing to draw. This happens "
            "when every aggregated communication strength was zero; check the "
            "gene programs and the group key passed to "
            "´compute_communication_gp_network´.")

    edge_types = np.unique(network_df["edge_type"])

    # One color per gene program, resolved once so that the drawing and the
    # legend can never disagree: both read this dict.
    if edge_type_colors is None:
        # Colorblindness adjusted vega_20
        # See https://github.com/theislab/scanpy/issues/387
        palette = list(map(colors.to_hex, cm.tab20.colors))
        palette[4] = "#279e68"  # green
        palette[8] = "#aa40fc"  # purple
        palette[16] = "#b5bd61"  # khaki
        edge_type_colors = palette
    if isinstance(edge_type_colors, dict):
        uncolored = [gp for gp in edge_types if gp not in edge_type_colors]
        if uncolored:
            raise ValueError(
                "´edge_type_colors´ does not cover the gene programs "
                f"{uncolored}.")
        edge_type_color_dict = dict(edge_type_colors)
    else:
        # ´zip´ would silently truncate here, leaving later gene programs
        # without a color and raising an opaque ´KeyError´ at draw time, so
        # cycle explicitly and say so.
        if len(edge_types) > len(edge_type_colors):
            warnings.warn(
                f"There are {len(edge_types)} gene programs but only "
                f"{len(edge_type_colors)} colors, so colors are reused and "
                "some gene programs are indistinguishable. Pass "
                "´edge_type_colors´ as a dict to control this.")
        edge_type_color_dict = {
            gp: edge_type_colors[i % len(edge_type_colors)]
            for i, gp in enumerate(edge_types)}

    G = nx.from_pandas_edgelist(
        network_df,
        source="source",
        target="target",
        edge_attr=["edge_type", edge_attr],
        create_using=nx.DiGraph())

    # The layout is derived from the GRAPH, not from ´adata´. Sizing the circle
    # by the number of categories while placing the graph's nodes on it leaves
    # the nodes on a fraction of the circle when a category was filtered out,
    # which reads as if the missing categories had been merged away rather
    # than dropped.
    if reserve_all_categories:
        if cat_key not in adata.obs:
            raise ValueError(f"´cat_key´ {cat_key!r} is not a column of "
                             "´adata.obs´.")
        G.add_nodes_from(adata.obs[cat_key].unique().tolist())
    node_list = sorted(G.nodes())
    n_nodes = len(node_list)

    uncolored_nodes = [node for node in node_list if node not in cat_colors]
    if uncolored_nodes:
        raise ValueError(
            f"´cat_colors´ does not cover the categories {uncolored_nodes}. "
            "Without this the colors and the nodes go out of step and "
            "matplotlib raises about the length of its ´c´ argument instead.")

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.axis("off")

    angle_dict = {node: 2.0 * np.pi * i / n_nodes
                  for i, node in enumerate(node_list)}
    pos = {node: (np.cos(theta), np.sin(theta))
           for node, theta in angle_dict.items()}
    node_color = [cat_colors[node] for node in node_list]

    # Partition the edges once. Self loops are drawn separately because they
    # need ´arrows=False´ to render as a loop rather than a zero length arrow.
    self_loops, other_edges = [], []
    for u, v, edge_data in G.edges(data=True):
        (self_loops if u == v else other_edges).append((u, v, edge_data))

    def _draw(edges, arrows):
        if not edges:
            return
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in edges],
            width=[d[edge_attr] * edge_width_scale for _, _, d in edges],
            edge_color=[edge_type_color_dict[d["edge_type"]]
                        for _, _, d in edges],
            node_size=node_size,
            arrows=arrows,
            ax=ax,
            # networkx warns that arrow styling is ignored when ´arrows´ is
            # False, which is the self loop pass, so only the arrow pass gets
            # it. Self loops render as loops either way.
            **({"arrowstyle": "-|>",
                "arrowsize": 20,
                "connectionstyle": connection_style} if arrows else {}))

    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=node_list,
                           node_size=node_size,
                           node_color=node_color,
                           ax=ax)
    _draw(other_edges, arrows=True)
    _draw(self_loops, arrows=False)

    # Labels are placed outside the circle and rotated to point outwards, so
    # they need the rendered text extent, which only exists after a draw.
    description = nx.draw_networkx_labels(G, pos, font_size=fontsize, ax=ax)
    fig.canvas.draw()
    trans = ax.transData.inverted()
    renderer = fig.canvas.get_renderer()
    for node, text in description.items():
        bbox = text.get_window_extent(renderer=renderer).transformed(trans)
        radius = text_space + bbox.width / 2.0
        theta = angle_dict[node]
        text.set_position((radius * np.cos(theta), radius * np.sin(theta)))
        text.set_rotation(np.degrees(theta))
        text.set_clip_on(False)

    if plot_legend:
        # Colors and labels come from the same ordered source. Building the
        # handles from ´set(edge_colors)´ instead pairs them by set iteration
        # order, which is salted per process, so the legend differed between
        # runs of the same script and silently dropped a gene program whose
        # only surviving edge was a self loop.
        drawn = [gp for gp in edge_types
                 if gp in {d["edge_type"] for _, _, d in G.edges(data=True)}]
        handles = [Line2D([0, 1], [0, 1],
                          color=edge_type_color_dict[gp], lw=5)
                   for gp in drawn]
        labels = [gp if gp.endswith("GP") else f"{gp} GP" for gp in drawn]
        ax.legend(handles, labels, loc="lower left")

    fig.tight_layout()
    if save:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return ax
