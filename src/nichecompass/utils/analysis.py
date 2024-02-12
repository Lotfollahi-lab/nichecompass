"""
This module contains utilities to analyze niches inferred by the NicheCompass
model.
"""

from typing import Optional, Tuple

#import holoviews as hv
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
    neighbor_cell_type_index = adata.obs[cell_type_key][nz_obsp_idx[1]].map(
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
                                    spot_size: float=30.):
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
        if n_top_peaks_per_gp > 0:
            adata_atac.X = adata_atac.X.toarray()
        adata.uns["omics_ft_pos_cmap"] = "Blues"
        adata.uns["omics_ft_neg_cmap"] = "Reds"
        
    cats = list(adata.uns[differential_gp_test_results_key]["category"][
        n_top_enriched_gp_start_idx:n_top_enriched_gp_end_idx])
    gps = list(adata.uns[differential_gp_test_results_key]["gene_program"][
        n_top_enriched_gp_start_idx:n_top_enriched_gp_end_idx])
    log_bayes_factors = list(adata.uns[differential_gp_test_results_key]["log_bayes_factor"][
        n_top_enriched_gp_start_idx:n_top_enriched_gp_end_idx])
    
    for gp in gps:
        # Get source and target genes, gene importances and gene signs and store
        # in temporary adata
        gp_gene_importances_df = model.compute_gp_gene_importances(
            selected_gp=gp)
        
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
                selected_gp=gp)
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
                title=f"{gp[:gp.index('_')]}\n"
                      f"{gp[gp.index('_') + 1: gp.rindex('_')].replace('_', ' ')}"
                      f"\n{gp[gps[i].rindex('_') + 1:]} score (LBF: {round(log_bayes_factors[i])})",
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
                title=f"{gps[i].split('_', 1)[0]}\n{gps[i].split('_', 1)[1]} "
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
    "41": "#4169E1", # Royal Blue
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
    "77": "#C38D9E", # Mauve-Pink
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
    new_categories = adata.obs[cat_key].unique().tolist()
    if color_palette == "cell_type_30":
        # https://github.com/scverse/scanpy/blob/master/scanpy/plotting/palettes.py#L40
        new_color_dict = {key: value for key, value in zip(
            new_categories,
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
             "#336600"])}
    elif color_palette == "cell_type_20":
        # https://github.com/vega/vega/wiki/Scales#scale-range-literals (some adjusted)
        new_color_dict = {key: value for key, value in zip(
            new_categories,
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
             '#8c6d31'])}
    elif color_palette == "cell_type_10":
        # scanpy vega10
        new_color_dict = {key: value for key, value in zip(
            new_categories,
            ['#7f7f7f',
             '#ff7f0e',
             '#279e68',
             '#e377c2',
             '#17becf',
             '#8c564b',
             '#d62728',
             '#1f77b4',
             '#b5bd61',
             '#aa40fc'])}
    elif color_palette == "batch":
        # sns.color_palette("colorblind").as_hex()
        new_color_dict = {key: value for key, value in zip(
            new_categories,
            ['#0173b2', '#d55e00', '#ece133', '#ca9161', '#fbafe4',
             '#949494', '#de8f05', '#029e73', '#cc78bc', '#56b4e9',
             '#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF',
             '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF',
             '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
             '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C'])}
    elif color_palette == "default":
        new_color_dict = {key: value for key, value in zip(new_categories, list(default_color_dict.values())[skip_default_colors:])}
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


def compute_communication_gp_network(
    gp_list: list,
    model: NicheCompass,
    group_key: str="niche",
    filter_key: Optional[str]=None,
    filter_cat: Optional[str]=None,
    n_neighbors: int=90):
    """
    Compute a network of category aggregated cell-pair communication strengths.
    
    First, compute cell-cell communication potential scores for each cell.
    Then dot product them and take into account neighborhoods to compute
    cell-pair communication strengths. Then, normalize cell-pair communication
    strengths.
    
    Parameters
    ----------
    gp_list:
        List of GPs for which the cell-pair communication strengths are computed.
    model:
        A trained NicheCompass model.
    group_key:
        Key in ´adata.obs´ where the groups are stored over which the cell-pair
        communication strengths will be aggregated.
    filter_key:
        Key in ´adata.obs´ that contains the category for which the results are
        filtered.
    filter_cat:
        Category for which the results are filtered.
    n_neighbors:
        Number of neighbors for the gp-specific neighborhood graph.

    Returns
    ----------
    network_df:
        A pandas dataframe with aggregated, normalized cell-pair communication strengths.
    """
    # Compute neighborhood graph
    compute_knn = True
    if 'spatial_cci' in model.adata.uns.keys():
        if model.adata.uns['spatial_cci']['params']['n_neighbors'] == n_neighbors:
            compute_knn = False
    if compute_knn:
        sc.pp.neighbors(model.adata,
                        n_neighbors=n_neighbors,
                        use_rep="spatial",
                        key_added="spatial_cci")
    
    gp_network_dfs = []
    gp_summary_df = model.get_gp_summary()
    for gp in gp_list:
        gp_idx = model.adata.uns[model.gp_names_key_].tolist().index(gp)
        active_gp_idx = model.adata.uns[model.active_gp_names_key_].tolist().index(gp)
        gp_scores = model.adata.obsm[model.latent_key_][:, active_gp_idx]
        gp_targets_cats = model.adata.varm[model.gp_targets_categories_mask_key_][:, gp_idx]
        gp_sources_cats = model.adata.varm[model.gp_sources_categories_mask_key_][:, gp_idx]
        targets_cats_label_encoder = model.adata.uns[model.targets_categories_label_encoder_key_]
        sources_cats_label_encoder = model.adata.uns[model.sources_categories_label_encoder_key_]

        sources_cat_idx_dict = {}
        for source_cat, source_cat_label in sources_cats_label_encoder.items():
            sources_cat_idx_dict[source_cat] = np.where(gp_sources_cats == source_cat_label)[0]

        targets_cat_idx_dict = {}
        for target_cat, target_cat_label in targets_cats_label_encoder.items():
            targets_cat_idx_dict[target_cat] = np.where(gp_targets_cats == target_cat_label)[0]

        # Get indices of all source and target genes
        source_genes_idx = np.array([], dtype=np.int64)
        for key in sources_cat_idx_dict.keys():
            source_genes_idx = np.append(source_genes_idx,
                                         sources_cat_idx_dict[key])
        target_genes_idx = np.array([], dtype=np.int64)
        for key in targets_cat_idx_dict.keys():
            target_genes_idx = np.append(target_genes_idx,
                                         targets_cat_idx_dict[key])

        # Compute cell-cell communication potential scores
        gp_source_scores = np.zeros((len(model.adata.obs), len(source_genes_idx)))
        gp_target_scores = np.zeros((len(model.adata.obs), len(target_genes_idx)))

        for i, source_gene_idx in enumerate(source_genes_idx):
            source_gene = model.adata.var_names[source_gene_idx]
            gp_source_scores[:, i] = (
                model.adata[:, model.adata.var_names.tolist().index(source_gene)].X.toarray().flatten() / model.adata[:, model.adata.var_names.tolist().index(source_gene)].X.toarray().flatten().max() *
                gp_summary_df[gp_summary_df["gp_name"] == gp]["gp_source_genes_weights"].values[0][gp_summary_df[gp_summary_df["gp_name"] == gp]["gp_source_genes"].values[0].index(source_gene)] *
                gp_scores)

        for j, target_gene_idx in enumerate(target_genes_idx):
            target_gene = model.adata.var_names[target_gene_idx]
            gp_target_scores[:, j] = (
                model.adata[:, model.adata.var_names.tolist().index(target_gene)].X.toarray().flatten() / model.adata[:, model.adata.var_names.tolist().index(target_gene)].X.toarray().flatten().max() *
                gp_summary_df[gp_summary_df["gp_name"] == gp]["gp_target_genes_weights"].values[0][gp_summary_df[gp_summary_df["gp_name"] == gp]["gp_target_genes"].values[0].index(target_gene)] *
                gp_scores)

        agg_gp_source_score = gp_source_scores.mean(1).astype("float32")
        agg_gp_target_score = gp_target_scores.mean(1).astype("float32")
        agg_gp_source_score[agg_gp_source_score < 0] = 0.
        agg_gp_target_score[agg_gp_target_score < 0] = 0.

        model.adata.obs[f"{gp}_source_score"] = agg_gp_source_score
        model.adata.obs[f"{gp}_target_score"] = agg_gp_target_score
        
        del(gp_target_scores)
        del(gp_source_scores)

        agg_gp_source_score = sp.csr_matrix(agg_gp_source_score)
        agg_gp_target_score = sp.csr_matrix(agg_gp_target_score)

        model.adata.obsp[f"{gp}_connectivities"] = (model.adata.obsp["spatial_cci_connectivities"] > 0).multiply(
            agg_gp_source_score.T.dot(agg_gp_target_score))

        # Aggregate gp connectivities for each group
        gp_network_df_pivoted = aggregate_obsp_matrix_per_cell_type(
            adata=model.adata,
            obsp_key=f"{gp}_connectivities",
            cell_type_key=group_key,
            group_key=filter_key,
            agg_rows=True)

        if filter_key is not None:
            gp_network_df_pivoted = gp_network_df_pivoted.loc[filter_cat, :]

        gp_network_df = gp_network_df_pivoted.melt(var_name="source", value_name="gp_score", ignore_index=False).reset_index()
        gp_network_df.columns = ["source", "target", "strength"]

        gp_network_df = gp_network_df.sort_values("strength", ascending=False)

        # Normalize strength
        min_value = gp_network_df["strength"].min()
        max_value = gp_network_df["strength"].max()
        gp_network_df["strength_unscaled"] = gp_network_df["strength"]
        gp_network_df["strength"] = (gp_network_df["strength"] - min_value) / (max_value - min_value)
        gp_network_df["strength"] = np.round(gp_network_df["strength"], 2)
        gp_network_df = gp_network_df[gp_network_df["strength"] > 0]

        gp_network_df["edge_type"] = gp
        gp_network_dfs.append(gp_network_df)

    network_df = pd.concat(gp_network_dfs, ignore_index=True)
    return network_df


def visualize_communication_gp_network(
    adata,
    network_df,
    cat_colors,
    edge_type_colors: Optional[dict]=None,
    edge_width_scale: int=20.0,
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
    edge_attr: str="strength"):
    """
    Visualize a communication gp network.
    """
    # Assuming you have unique edge types in your 'edge_type' column
    edge_types = np.unique(network_df['edge_type'])
    
    if edge_type_colors is None:
        # Colorblindness adjusted vega_10
        # See https://github.com/theislab/scanpy/issues/387
        vega_10 = list(map(colors.to_hex, cm.tab10.colors))
        vega_10_scanpy = vega_10.copy()
        vega_10_scanpy[2] = "#279e68"  # green
        vega_10_scanpy[4] = "#aa40fc"  # purple
        vega_10_scanpy[8] = "#b5bd61"  # kakhi
        edge_type_colors = vega_10_scanpy

    # Create a dictionary that maps edge types to colors
    edge_type_color_dict = {edge_type: color for edge_type, color in zip(edge_types, edge_type_colors)}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.axis("off")
    G = nx.from_pandas_edgelist(
        network_df,
        source="source",
        target="target",
        edge_attr=["edge_type", edge_attr],
        create_using=nx.DiGraph(),
    )
    pos = nx.circular_layout(G)

    nx.set_node_attributes(G, cat_colors, "color")
    node_color = nx.get_node_attributes(G, "color")

    description = nx.draw_networkx_labels(G, pos, font_size=fontsize)
    n = adata.obs[cat_key].nunique()
    node_list = sorted(G.nodes())
    angle = []
    angle_dict = {}
    for i, node in zip(range(n), node_list):
        theta = 2.0 * np.pi * i / n
        angle.append((np.cos(theta), np.sin(theta)))
        angle_dict[node] = theta
    pos = {}
    for node_i, node in enumerate(node_list):
        pos[node] = angle[node_i]

    r = fig.canvas.get_renderer()
    trans = plt.gca().transData.inverted()
    for node, t in description.items():
        bb = t.get_window_extent(renderer=r)
        bbdata = bb.transformed(trans)
        radius = text_space + bbdata.width / 2.0
        position = (radius * np.cos(angle_dict[node]), radius * np.sin(angle_dict[node]))
        t.set_position(position)
        t.set_rotation(angle_dict[node] * 360.0 / (2.0 * np.pi))
        t.set_clip_on(False)

    edgelist = [(u, v) for u, v, e in G.edges(data=True) if u != v]
    edge_colors = [edge_type_color_dict[edge_data['edge_type']] for u, v, edge_data in G.edges(data=True) if u != v]
    width = [e[edge_attr] * edge_width_scale for u, v, e in G.edges(data=True) if u != v]

    h2 = nx.draw_networkx(
        G,
        pos,
        with_labels=False,
        node_size=node_size,
        edgelist=edgelist,
        width=width,
        edge_vmin=0.0,
        edge_vmax=1.0,
        edge_color=edge_colors,  # Use the edge type colors here
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        vmin=0.0,
        vmax=1.0,
        cmap=plt.cm.binary,  # Use a colormap for node colors if needed
        node_color=list(node_color.values()),
        ax=ax,
        connectionstyle=connection_style,
    )

    #https://stackoverflow.com/questions/19877666/add-legends-to-linecollection-plot - uses plotted data to define the color but here we already have colors defined, so just need a Line2D object.
    def make_proxy(clr, mappable, **kwargs):
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)

    # generate proxies with the above function
    proxies = [make_proxy(clr, h2, lw=5) for clr in set(edge_colors)]
    labels = [edge.split("_")[0] + " GP" for edge in edge_types[::-1]]

    if plot_legend:
        lgd = plt.legend(proxies, labels, loc="lower left")

    edgelist = [(u, v) for u, v, e in G.edges(data=True) if ((u == v))] + [(u, v) for u, v, e in G.edges(data=True) if ((u != v))]
    edge_colors = [edge_type_color_dict[edge_data['edge_type']] for u, v, edge_data in G.edges(data=True) if u == v]
    width = [e[edge_attr] * edge_width_scale for u, v, e in G.edges(data=True) if u == v] + [0 for u, v, e in G.edges(data=True) if ((u != v))]
    nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_size,
        edgelist=edgelist, 
        width=width,
        edge_vmin=0.0,
        edge_vmax=1.0,
        edge_color=edge_colors,
        arrows=False,
        arrowstyle="-|>",
        arrowsize=20,
        ax=ax,
        connectionstyle=connection_style)
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    plt.ion()
