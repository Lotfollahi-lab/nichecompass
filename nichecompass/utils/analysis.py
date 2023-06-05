"""
This module contains utilities to analyze niches inferred by the NicheCompass
model.
"""

from typing import Optional

import holoviews as hv
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


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
        save_path: Optional[str]=None):
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
    save_path:
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
                save_path,
                fmt="png")

        
def generate_gp_info_plots(analysis_label,
                           differential_gp_test_results_key,
                           model,
                           cell_type_key,
                           cell_type_colors,
                           latent_cluster_colors,
                           plot_category,
                           log_bayes_factor_thresh,
                           n_top_enriched_gps=10,
                           adata=None,
                           feature_spaces=["latent", "physical_embryo1", "physical_embryo2", "physical_embryo3"],
                           plot_types=["gene_categories", "top_genes"],
                           n_top_genes_per_gp=3,
                           save_figs=False,
                           figure_folder_path="",
                           spot_size=30):
    
    if adata is None:
        adata = model.adata.copy()
        
    cats = adata.uns[differential_gp_test_results_key]["category"][:n_top_enriched_gps]
    gps = adata.uns[differential_gp_test_results_key]["gene_program"][:n_top_enriched_gps]
    
    for gp in gps:
        gp_gene_importances_df = model.compute_gp_gene_importances(selected_gp=gp)
        
        if "gene_categories" in plot_types:
            pos_sign_target_genes = gp_gene_importances_df.loc[
                (gp_gene_importances_df["gene_weight_sign_corrected"] > 0) &
                (gp_gene_importances_df["gene_entity"] == "target"), "gene"].tolist()
            pos_sign_source_genes = gp_gene_importances_df.loc[
                (gp_gene_importances_df["gene_weight_sign_corrected"] > 0) &
                (gp_gene_importances_df["gene_entity"] == "source"), "gene"].tolist()
            neg_sign_target_genes = gp_gene_importances_df.loc[
                (gp_gene_importances_df["gene_weight_sign_corrected"] < 0) &
                (gp_gene_importances_df["gene_entity"] == "target"), "gene"].tolist()
            neg_sign_source_genes = gp_gene_importances_df.loc[
                (gp_gene_importances_df["gene_weight_sign_corrected"] < 0) &
                (gp_gene_importances_df["gene_entity"] == "source"), "gene"].tolist()

            pos_sign_target_gene_importances = gp_gene_importances_df.loc[
                (gp_gene_importances_df["gene_weight_sign_corrected"] > 0) &
                (gp_gene_importances_df["gene_entity"] == "target"), "gene_importance"].values.reshape(1, -1)
            pos_sign_source_gene_importances = gp_gene_importances_df.loc[
                (gp_gene_importances_df["gene_weight_sign_corrected"] > 0) &
                (gp_gene_importances_df["gene_entity"] == "source"), "gene_importance"].values.reshape(1, -1)
            neg_sign_target_gene_importances = gp_gene_importances_df.loc[
                (gp_gene_importances_df["gene_weight_sign_corrected"] < 0) &
                (gp_gene_importances_df["gene_entity"] == "target"), "gene_importance"].values.reshape(1, -1)
            neg_sign_source_gene_importances = gp_gene_importances_df.loc[
                (gp_gene_importances_df["gene_weight_sign_corrected"] < 0) &
                (gp_gene_importances_df["gene_entity"] == "source"), "gene_importance"].values.reshape(1, -1)

            pos_sign_target_gene_expr = adata[:, pos_sign_target_genes].X.toarray()
            pos_sign_source_gene_expr = adata[:, pos_sign_source_genes].X.toarray()
            neg_sign_target_gene_expr = adata[:, neg_sign_target_genes].X.toarray()
            neg_sign_source_gene_expr = adata[:, neg_sign_source_genes].X.toarray()

            adata.obs[f"{gp}_pos_sign_target_gene_weighted_mean_gene_expr"] = (
                np.mean(pos_sign_target_gene_expr * pos_sign_target_gene_importances, axis=1))
            adata.obs[f"{gp}_pos_sign_source_gene_weighted_mean_gene_expr"] = (
                np.mean(pos_sign_source_gene_expr * pos_sign_source_gene_importances, axis=1))
            adata.obs[f"{gp}_neg_sign_target_gene_weighted_mean_gene_expr"] = (
                np.mean(neg_sign_target_gene_expr * neg_sign_target_gene_importances, axis=1))
            adata.obs[f"{gp}_neg_sign_source_gene_weighted_mean_gene_expr"] = (
                np.mean(neg_sign_source_gene_expr * neg_sign_source_gene_importances, axis=1))

            adata.uns[f"{gp}_gene_category_importances"] = np.array([pos_sign_target_gene_importances.sum(),
                                                                     pos_sign_source_gene_importances.sum(),
                                                                     neg_sign_target_gene_importances.sum(),
                                                                     neg_sign_source_gene_importances.sum()])
        
        if "top_genes" in plot_types:
            gp_source_genes_gene_importances_df = gp_gene_importances_df[
                gp_gene_importances_df["gene_entity"] == "source"]
            
            gp_target_genes_gene_importances_df = gp_gene_importances_df[
                gp_gene_importances_df["gene_entity"] == "target"]
            
            adata.uns["n_top_source_genes"] = n_top_genes_per_gp
            adata.uns[f"{gp}_source_genes_top_genes"] = gp_source_genes_gene_importances_df["gene"][:n_top_genes_per_gp]
            adata.uns[f"{gp}_source_genes_top_gene_importances"] = gp_source_genes_gene_importances_df["gene_importance"][:n_top_genes_per_gp]
            adata.uns[f"{gp}_source_genes_top_gene_signs"] = np.where(gp_source_genes_gene_importances_df["gene_weight_sign_corrected"] > 0, "+", "-")
            adata.uns["n_top_target_genes"] = n_top_genes_per_gp
            adata.uns[f"{gp}_target_genes_top_genes"] = gp_target_genes_gene_importances_df["gene"][:n_top_genes_per_gp]
            adata.uns[f"{gp}_target_genes_top_gene_importances"] = gp_target_genes_gene_importances_df["gene_importance"][:n_top_genes_per_gp]
            adata.uns[f"{gp}_target_genes_top_gene_signs"] = np.where(gp_target_genes_gene_importances_df["gene_weight_sign_corrected"] > 0, "+", "-")
            
            #adata.uns["n_top_genes"] = n_top_genes_per_gp
            #adata.uns[f"{gp}_top_genes"] = gp_gene_importances_df["gene"][:n_top_genes_per_gp]
            #adata.uns[f"{gp}_top_gene_importances"] = gp_gene_importances_df["gene_importance"][:n_top_genes_per_gp]
            #adata.uns[f"{gp}_top_gene_signs"] = np.where(gp_gene_importances_df["gene_weight_sign_corrected"] > 0, "+", "-")
            #adata.uns[f"{gp}_top_gene_entities"] = gp_gene_importances_df["gene_entity"]
        
    for feature_space in feature_spaces:
        for plot_type in plot_types:
            plot_gp_info_plots(adata=adata,
                               cell_type_key=cell_type_key,
                               cell_type_colors=cell_type_colors,
                               latent_cluster_colors=latent_cluster_colors,
                               cats=cats,
                               gps=gps,
                               plot_type=plot_type,
                               plot_category=plot_category,
                               feature_space=feature_space,
                               spot_size=spot_size,
                               suptitle=f"{analysis_label.replace('_', ' ').title()} Top {n_top_enriched_gps} Enriched GPs: "
                                        f"GP Scores and {'Weighted Mean ' if plot_type == 'gene_categories' else ''}"
                                        f"Gene Expression of {plot_type.replace('_', ' ').title()} in {feature_space.replace('_', ' ').title()} Feature Space",
                               cat_title=f"Enriched GP Category in \n {plot_category.replace('_', ' ').title()}",
                               save_fig=save_figs,
                               figure_folder_path=figure_folder_path,
                               fig_name=f"{analysis_label}_log_bayes_factor_{log_bayes_factor_thresh}_enriched_gps_gp_scores_" \
                                        f"{'weighted_mean_gene_expr' if plot_type == 'gene_categories' else 'top_genes_gene_expr'}_" \
                                        f"{feature_space}_space")
            
def plot_gp_info_plots(adata,
                       cell_type_key,
                       cell_type_colors,
                       latent_cluster_colors,
                       cats,
                       gps,
                       plot_type,
                       plot_category,
                       feature_space,
                       spot_size,
                       suptitle,
                       cat_title,
                       save_fig,
                       figure_folder_path,
                       fig_name):
    if plot_category == cell_type_key:
        palette = cell_type_colors
    else:
        palette = latent_cluster_colors 
    # Plot selected gene program latent scores
    if plot_type == "gene_categories":
        ncols = 6
        fig_width = 36
        wspace = 0.155
    elif plot_type == "top_genes":
        ncols = 2 + adata.uns["n_top_genes"]
        fig_width = 12 + (6 * adata.uns["n_top_genes"])
        wspace = 0.3
    fig, axs = plt.subplots(nrows=len(gps), ncols=ncols, figsize=(fig_width, 6*len(gps)))
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)

    title = fig.suptitle(t=suptitle,
                         x=0.55,
                         y=(1.1 if len(gps) == 1 else 0.93),
                         fontsize=20)
    for i, gp in enumerate(gps):
        if feature_space == "latent":
            sc.pl.umap(adata,
                       color=plot_category,
                       palette=palette,
                       groups=cats[i],
                       ax=axs[i, 0],
                       title=cat_title,
                       legend_loc="on data",
                       na_in_legend=False,
                       show=False)
            sc.pl.umap(adata,
                       color=gps[i],
                       color_map="RdBu",
                       ax=axs[i, 1],
                       title=f"{gp[:gp.index('_')]}\n{gp[gp.index('_') + 1: gp.rindex('_')].replace('_', ' ')}\n{gp[gps[i].rindex('_') + 1:]} score",
                       show=False)
        elif "physical" in feature_space:
            sc.pl.spatial(adata=adata[adata.obs["batch"] == feature_space.split("_")[1]],
                          color=plot_category,
                          palette=palette,
                          groups=cats[i],
                          ax=axs[i, 0],
                          spot_size=spot_size,
                          title=cat_title,
                          legend_loc="on data",
                          na_in_legend=False,
                          show=False)
            sc.pl.spatial(adata=adata[adata.obs["batch"] == feature_space.split("_")[1]],
                          color=gps[i],
                          color_map="RdBu",
                          spot_size=spot_size,
                          title=f"{gps[i].split('_', 1)[0]}\n{gps[i].split('_', 1)[1]}",
                          legend_loc=None,
                          ax=axs[i, 1],
                          show=False) 
        axs[i, 0].xaxis.label.set_visible(False)
        axs[i, 0].yaxis.label.set_visible(False)
        axs[i, 1].xaxis.label.set_visible(False)
        axs[i, 1].yaxis.label.set_visible(False)
        if plot_type == "gene_categories":
            for j, gene_category in enumerate(["pos_sign_target_gene",
                                               "pos_sign_source_gene",
                                               "neg_sign_target_gene",
                                               "neg_sign_source_gene"]):
                if not adata.obs[f"{gp}_{gene_category}_weighted_mean_gene_expr"].isna().any():
                    if feature_space == "latent":
                        sc.pl.umap(adata,
                                   color=f"{gp}_{gene_category}_weighted_mean_gene_expr",
                                   color_map=("Blues" if "pos_sign" in gene_category else "Reds"),
                                   ax=axs[i, j+2],
                                   legend_loc="on data",
                                   na_in_legend=False,
                                   title=f"Weighted mean gene expression \n {gene_category.replace('_', ' ')} ({adata.uns[f'{gp}_gene_category_importances'][j]:.2f})",
                                   show=False)
                    elif "physical" in feature_space:
                        sc.pl.spatial(adata=adata[adata.obs["sample"] == feature_space.split("_")[1]],
                                      color=f"{gp}_{gene_category}_weighted_mean_gene_expr",
                                      color_map=("Blues" if "pos_sign" in gene_category else "Reds"),
                                      ax=axs[i, 2+j],
                                      legend_loc="on data",
                                      na_in_legend=False,
                                      groups=cats[i],
                                      spot_size=spot_size,
                                      title=f"Weighted mean gene expression \n {gene_category.replace('_', ' ')} ({adata.uns[f'{gp}_gene_category_importances'][j]:.2f})",
                                      show=False)                        
                    axs[i, j+2].xaxis.label.set_visible(False)
                    axs[i, j+2].yaxis.label.set_visible(False)
                else:
                    axs[i, j+2].set_visible(False)
        elif plot_type == "top_genes":
            for j in range(len(adata.uns[f"{gp}_top_genes"])):
                if feature_space == "latent":
                    sc.pl.umap(adata,
                               color=adata.uns[f"{gp}_top_genes"][j],
                               color_map=("Blues" if adata.uns[f"{gp}_top_gene_signs"][j] == "+" else "Reds"),
                               ax=axs[i, 2+j],
                               legend_loc="on data",
                               na_in_legend=False,
                               title=f"{adata.uns[f'{gp}_top_genes'][j]}: "
                                     f"{adata.uns[f'{gp}_top_gene_importances'][j]:.2f} "
                                     f"({adata.uns[f'{gp}_top_gene_entities'][j][0]}; "
                                     f"{adata.uns[f'{gp}_top_gene_signs'][j]})",
                               show=False)
                elif "physical" in feature_space:
                    sc.pl.spatial(adata=adata[adata.obs["batch"] == feature_space.split("_")[1]],
                                  color=adata.uns[f"{gp}_top_genes"][j],
                                  color_map=("Blues" if adata.uns[f"{gp}_top_gene_signs"][j] == "+" else "Reds"),
                                  legend_loc="on data",
                                  na_in_legend=False,
                                  ax=axs[i, 2+j],
                                  # groups=cats[i],
                                  spot_size=spot_size,
                                  title=f"{adata.uns[f'{gp}_top_genes'][j]}: "
                                        f"{adata.uns[f'{gp}_top_gene_importances'][j]:.2f} "
                                        f"({adata.uns[f'{gp}_top_gene_entities'][j][0]}; "
                                        f"{adata.uns[f'{gp}_top_gene_signs'][j]})",
                                  show=False)
                axs[i, 2+j].xaxis.label.set_visible(False)
                axs[i, 2+j].yaxis.label.set_visible(False)
            for k in range(len(adata.uns[f"{gp}_top_genes"]), ncols - 2):
                axs[i, 2+k].set_visible(False)

    # Save and display plot
    plt.subplots_adjust(wspace=wspace, hspace=0.275)
    if save_fig:
        fig.savefig(f"{figure_folder_path}/{fig_name}.svg",
                    bbox_extra_artists=(title,),
                    bbox_inches="tight")
    plt.show()