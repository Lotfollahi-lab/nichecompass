"""
This module contains helper functions for the ´utils´ subpackage.
"""

import os
from typing import Optional

import pandas as pd
import pyreadr


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