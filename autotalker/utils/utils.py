"""
This module contains helper functions for the ´utils´ subpackage.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import seaborn as sns


def load_R_file_as_df(R_file_path: str,
                       url: Optional[str]=None,
                       save_df_to_disk: bool=False,
                       df_save_path: Optional[str]=None) -> pd.DataFrame:
    """
    Helper to load an R file either from ´url´ if specified or from ´file_path´ 
    on disk and convert it to a pandas DataFrame.

    Parameters
    ----------
    R_file_path:
        File path to the R file to be loaded as df.
    url:
        URL of the R file to be loaded as df.
    save_df_to_disk:
        If ´True´, save df to disk.
    df_save_path:
        Path where the df will be saved if ´save_df_to_disk´ is ´True´.

    Returns
    ----------
    df:
        Content of R file loaded into a pandas DataFrame.
    """
    if url is None:
        if not os.path.exists(R_file_path):
            raise ValueError("Please specify a valid ´file_path´ or ´url´.")
        result_odict = pyreadr.read_r(R_file_path)
    else:
        result_odict = pyreadr.read_r(pyreadr.download_file(url, R_file_path))
        os.remove(R_file_path)

    df = result_odict[None]

    if save_df_to_disk:
        if df_save_path == None:
            raise ValueError("Please specify ´df_save_path´ or set " 
                             "´save_to_disk.´ to False.")
        df.to_csv(df_save_path)
    return df


def create_gp_gene_count_distribution_plots(gp_dict: dict,
                                            max_n_genes=100):
    """
    Create distribution plots of the gene counts for source and target genes
    of a gene program.
    """
    n_source_genes_list = []
    n_target_genes_list = []

    for _, gp_sources_targets_dict in gp_dict.items():
        n_source_genes_list.append(len(gp_sources_targets_dict["sources"]))
        n_target_genes_list.append(len(gp_sources_targets_dict["targets"]))

    # Convert the array to a pandas DataFrame
    target_genes_df = pd.DataFrame({"values": n_target_genes_list})
    source_genes_df = pd.DataFrame({"values": n_source_genes_list})

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    sns.countplot(x="values", data=target_genes_df, ax=ax1)
    ax1.set(title="Gene Program Target Genes Distribution",
            xlabel="Number of Target Genes",
            ylabel="Gene Program Count")
    ax1.set_xticks(np.arange(0, max_n_genes+1, 10))
    sns.countplot(x="values", data=source_genes_df, ax=ax2)
    ax2.set(title="Gene Program Source Genes Distribution",
        xlabel="Number of Source Genes",
        ylabel="Gene Program Count")
    ax2.set_xticks(np.arange(0, max_n_genes+1, 10))
    plt.subplots_adjust(wspace=0.35)
    plt.show()