"""
This module contains utilities to analyze niches inferred by the NicheCompass
model.
"""

from typing import Optional

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

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
        List of groups that will be plotted. If ´all´, plot all groups
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
                                title=f"Group {group_label}"))
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
                                    n_top_enriched_gps: int=10,
                                    feature_spaces: list=["latent"],
                                    n_top_genes_per_gp: int=3,
                                    n_top_peaks_per_gp: int=0,
                                    log_norm_omics_features: bool=True,
                                    save_figs: bool=False,
                                    figure_folder_path: str="",
                                    spot_size: float=30.):
    """
    Generate info plots of enriched gene programs, showing the enriched
    category, the gp scores, as well as the counts (or log normalized counts) of
    the top genes and/or peaks in a specified feature space.
    
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
    n_top_enriched_gps:
        Number of top enriched gene programs for which to create info plots.
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
    log_norm_omics_features:
        If ´True´, log normalize genes and peaks before plotting.
    save_figs:
        If ´True´, save the figures.
    figure_folder_path:
        Folder path where the figures will be saved.
    spot_size:
        Spot size used for the spatial plots.
    """
    model._check_if_trained(warn=True)

    adata = model.adata.copy()
    if n_top_peaks_per_gp > 0:
        if "chrom_access" not in model.modalities_:
            raise ValueError("The model needs to be trained with ATAC data if"
                             "'n_top_peaks_per_gp' > 0.")
        adata_atac = model.adata_atac.copy()
    
    if log_norm_omics_features:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if n_top_peaks_per_gp > 0:
            sc.pp.normalize_total(adata_atac, target_sum=1e4)
            sc.pp.log1p(adata_atac)
        
    cats = adata.uns[differential_gp_test_results_key]["category"][
        :n_top_enriched_gps]
    gps = adata.uns[differential_gp_test_results_key]["gene_program"][
        :n_top_enriched_gps]
    
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
                "gene_weight_sign_corrected"] > 0, "+", "-"))
        adata.uns["n_top_target_genes"] = n_top_genes_per_gp
        adata.uns[f"{gp}_target_genes_top_genes"] = (
            gp_target_genes_gene_importances_df["gene"][
                :n_top_genes_per_gp].values)
        adata.uns[f"{gp}_target_genes_top_gene_importances"] = (
            gp_target_genes_gene_importances_df["gene_importance"][
                :n_top_genes_per_gp].values)
        adata.uns[f"{gp}_target_genes_top_gene_signs"] = (
            np.where(gp_target_genes_gene_importances_df[
                "gene_weight_sign_corrected"] > 0, "+", "-"))

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
                    "peak_weight_sign_corrected"] > 0, "+", "-"))
            adata.uns["n_top_target_peaks"] = n_top_peaks_per_gp
            adata.uns[f"{gp}_target_peaks_top_peaks"] = (
                gp_target_peaks_peak_importances_df["peak"][
                    :n_top_peaks_per_gp].values)
            adata.uns[f"{gp}_target_peaks_top_peak_importances"] = (
                gp_target_peaks_peak_importances_df["peak_importance"][
                    :n_top_peaks_per_gp].values)
            adata.uns[f"{gp}_target_peaks_top_peak_signs"] = (
                np.where(gp_target_peaks_peak_importances_df[
                    "peak_weight_sign_corrected"] > 0, "+", "-"))
            
            # Add peak counts to temporary adata for plotting
            adata.obs[[peak for peak in 
                       adata.uns[f"{gp}_target_peaks_top_peaks"]]] = (
                adata_atac.X.toarray()[
                    :, [adata_atac.var_names.tolist().index(peak)
                        for peak in adata.uns[f"{gp}_target_peaks_top_peaks"]]])
            adata.obs[[peak for peak in
                       adata.uns[f"{gp}_source_peaks_top_peaks"]]] = (
                adata_atac.X.toarray()[
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
            cat_key=cat_key,
            cat_palette=cat_palette,
            cats=cats,
            feature_space=feature_space,
            spot_size=spot_size,
            suptitle=f"{plot_label.replace('_', ' ').title()} "
                     f"Top {n_top_enriched_gps} Enriched GPs: "
                     f"GP Scores and Omics Feature Counts in "
                     f"{feature_space} Feature Space",
            save_fig=save_figs,
            figure_folder_path=figure_folder_path,
            fig_name=f"{plot_label}_top_enriched_gps_gp_scores_"
                     f"omics_feature_counts_in_{feature_space}_"
                     "feature_space")
            
            
def plot_enriched_gp_info_plots_(adata: AnnData,
                                 sample_key: str,
                                 gps: list,
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
                         y=(1.1 if len(gps) == 1 else 0.93),
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
                      f"\n{gp[gps[i].rindex('_') + 1:]} score",
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
                title=f"{gps[i].split('_', 1)[0]}\n{gps[i].split('_', 1)[1]}",
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
                        color_map=("Blues" if
                                   adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1][:-1]}"
                                             "_signs"][j] == "+" else "Reds"),
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
                        color_map=("Blues" if
                                   adata.uns[f"{gp}_{modality_entity}_top_"
                                             f"{modality_entity.split('_')[1][:-1]}"
                                             "_signs"][j] == "+" else "Reds"),
                        legend_loc="on data",
                        na_in_legend=False,
                        ax=axs[i, 2+k+j],
                        spot_size=spot_size,
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
        fig.savefig(f"{figure_folder_path}/{fig_name}.svg",
                    bbox_extra_artists=(title,),
                    bbox_inches="tight")
    plt.show()