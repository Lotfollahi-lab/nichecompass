"""
This module contains utilities to analyze cell interactions inferred by the
NicheCompass model.
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
    save_fig:
        If ´True´, save the figure.
    save_path:
        Path where to save the figure.
    """
    hv.extension("bokeh")
    hv.output(size=200)

    sorted_cell_types = sorted(adata.obs[cell_type_key].unique().tolist())

    # Get group labels
    if group_key is not None:
        group_labels = df.index.get_level_values(
            df.index.names.index(group_key)).unique().tolist()
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
