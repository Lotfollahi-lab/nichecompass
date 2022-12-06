"""
This module contains helper functions for the ´benchmarking´ subpackage.
"""

from typing import Optional

import numpy as np


def convert_to_one_hot(vector: np.ndarray,
                       n_classes: Optional[int]):
    """
    Converts an input 1D vector of integer labels into a 2D array of one-hot 
    vectors, where for an i'th input value of j, a '1' will be inserted in the 
    i'th row and j'th column of the output one-hot vector. Adapted from 
    https://github.com/theislab/scib/blob/29f79d0135f33426481f9ff05dd1ae55c8787142/scib/metrics/lisi.py#L498
    (05.12.22).

    Parameters
    ----------
    vector:
        Vector to be one-hot-encoded.
    n_classes:
        Number of classes to be considered for one-hot-encoding. If ´None´, the
        number of classes will be inferred from ´vector´.

    Returns
    ----------
    one_hot:
        2D NumPy array of one-hot-encoded vectors.

    Example:
    ´´´
    vector = np.array((1, 0, 4))
    one_hot = _convert_to_one_hot(vector)
    print(one_hot)
    [[0 1 0 0 0]
     [1 0 0 0 0]
     [0 0 0 0 1]]
    ´´´
    """
    if n_classes is None:
        n_classes = np.max(vector) + 1

    one_hot = np.zeros(shape=(len(vector), n_classes))
    one_hot[np.arange(len(vector)), vector] = 1
    return one_hot.astype(int)
