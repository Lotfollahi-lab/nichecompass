import os
from typing import Optional

import pyreadr


def _load_R_file_as_df(R_file_path: str,
                       url: Optional[str]=None,
                       save_df_to_disk: bool=False,
                       df_save_path: Optional[str]=None):
    """
    Helper to load an R file either from ´url´ if specified or from ´file_path´ 
    on disk and convert to a pandas DataFrame.
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
                             "´save_to_disk.´ to False")
        df.to_csv(df_save_path)

    return df