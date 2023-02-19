"""
This module contains utiilities to analyze cell interactions inferred by the
Autotalker model.
"""

import holoviews as hv
import numpy as np
import pandas as pd
from anndata import AnnData


def aggregate_node_label_agg_att_weights_per_cell_type(
        adata: AnnData,
        agg_alpha_key: str="autotalker_agg_alpha",
        cell_type_key: str="cell_type"):
    """
    Aggregate the node label aggregator attention weights alpha of a trained
    Autotalker model by neighbor cell type.

    Parameters
    ----------
    adata:
        AnnData object which contains outputs of Autotalker model training.
    agg_alpha_key:
        Key in ´adata.obsp´ where the attention weights of the gene expression
        node label aggregator are stored.
    cell_type_key:
        Key in ´adata.obs´ where the cell type labels are stored.

    Returns
    ---------- 
    """
    n_obs = len(adata)
    n_cell_types = adata.obs[cell_type_key].nunique()
    sorted_cell_types = sorted(adata.obs[cell_type_key].unique().tolist())

    cell_type_label_encoder = {k: v for k, v in zip(
        sorted_cell_types,
        range(n_cell_types))}

    nz_alpha_idx = adata.obsp[agg_alpha_key].nonzero()
    neighbor_cell_type_index = adata.obs[cell_type_key][nz_alpha_idx[1]].map(
        cell_type_label_encoder).values
    nz_alpha = adata.obsp[agg_alpha_key].data

    cell_type_agg_alpha = np.zeros((n_obs, n_cell_types))
    np.add.at(cell_type_agg_alpha,
              (nz_alpha_idx[0], neighbor_cell_type_index),
              nz_alpha)
    cell_type_agg_alpha_df = pd.DataFrame(
        cell_type_agg_alpha,
        columns=sorted_cell_types)
    cell_type_agg_alpha_df[cell_type_key] = adata.obs[cell_type_key].values
    return cell_type_agg_alpha_df


def create_cell_type_chord_plot_from_df(
        adata: AnnData,
        df: pd.DataFrame,
        link_threshold: float=0.1,
        cell_type_key: str="cell_type"):
    """
    Create a cell type chord diagram based on an input DataFrame.

    Parameters
    ----------
    adata:
        AnnData object which contains outputs of Autotalker model training.
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
                             node_color=hv.dim("index").str()))
    hv.output(chord)