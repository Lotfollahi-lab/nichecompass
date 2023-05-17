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
        agg_rows: bool=False):
    """
    Generic function to aggregate adjacency matrices stored in
    ´adata.obsp[obsp_key]´ on cell type level. It can be used to aggregate the
    node label aggregator attention weights alpha or the reconstructed adjacency
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

    nz_obsp_idx = adata.obsp[obsp_key].nonzero()
    neighbor_cell_type_index = adata.obs[cell_type_key][nz_obsp_idx[1]].map(
        cell_type_label_encoder).values
    adata.obsp[obsp_key].eliminate_zeros() # In some sparse reps 0s can appear
    nz_obsp = adata.obsp[obsp_key].data

    cell_type_agg = np.zeros((n_obs, n_cell_types))
    np.add.at(cell_type_agg,
              (nz_obsp_idx[0], neighbor_cell_type_index),
              nz_obsp)
    cell_type_agg_df = pd.DataFrame(
        cell_type_agg,
        columns=sorted_cell_types)
    cell_type_agg_df[cell_type_key] = adata.obs[cell_type_key].values

    if agg_rows:
        cell_type_agg_df = cell_type_agg_df.groupby(cell_type_key).sum()
    return cell_type_agg_df


def create_cell_type_chord_plot_from_df(
        adata: AnnData,
        df: pd.DataFrame,
        title: str="Cell Interactions",
        link_threshold: float=0.1,
        cell_type_key: str="cell_type",
        save_fig: bool=False,
        save_path: Optional[str]=None):
    """
    Create a cell type chord diagram based on an input DataFrame.

    Parameters
    ----------
    adata:
        AnnData object which contains outputs of NicheCompass model training.
    df:
        A Pandas DataFrame that contains the connection values for the chord
        plot (dim: n_cell_types x n_cell_types).
    link_threshold:
        Ratio of attention that a cell type needs to exceed compared to the cell
        type with the maximum attention to be considered a link for the chord
        plot.
    cell_type_key:
        Key in ´adata.obs´ where the cell type labels are stored.

    Returns
    ----------
    """
    hv.extension("bokeh")
    hv.output(size=200)

    sorted_cell_types = sorted(adata.obs[cell_type_key].unique().tolist())

    max_attention_values = df.max(axis=1).values

    links_list = []
    for i in range(len(sorted_cell_types)):
        for j in range(len(sorted_cell_types)):
            if df.iloc[i, j] > max_attention_values[i] * link_threshold:
                link_dict = {}
                link_dict["source"] = j
                link_dict["target"] = i
                link_dict["value"] = df.iloc[i, j]
                links_list.append(link_dict)
    links = pd.DataFrame(links_list)

    nodes_list = []
    for cell_type in sorted_cell_types:
        nodes_dict = {}
        nodes_dict["name"] = cell_type
        nodes_dict["group"] = 1
        nodes_list.append(nodes_dict)
    nodes = hv.Dataset(pd.DataFrame(nodes_list), "index")

    chord = hv.Chord((links, nodes)).select(value=(5, None))
    chord.opts(hv.opts.Chord(cmap="Category20",
                             edge_cmap="Category20",
                             edge_color=hv.dim("source").str(),
                             labels="name",
                             node_color=hv.dim("index").str(),
                             title=title))
    hv.output(chord)

    if save_fig:
        hv.save(chord,
                save_path,
                fmt="png")
