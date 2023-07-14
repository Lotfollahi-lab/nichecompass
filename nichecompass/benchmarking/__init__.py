# This is a trick to make jax use the right cudnn version (needs to be executed
# before importing scanpy)
import jax.numpy as jnp
temp_array = jnp.array([1, 2, 3])
temp_idx = jnp.array([1])
temp_array[temp_idx]

from .cas import compute_avg_cas, compute_cas
from .gcs import compute_avg_gcs, compute_gcs
from .cca import compute_cca
from .gerr2 import compute_gerr2
from .metrics import compute_benchmarking_metrics
from .mlami import compute_mlami
from .nasw import compute_nasw
from .clisis import compute_clisis

__all__ = ["compute_avg_cas",
           "compute_avg_gcs",
           "compute_benchmarking_metrics",
           "compute_cas",
           "compute_cca",
           "compute_gcs",
           "compute_gerr2",
           "compute_mlami",
           "compute_nasw",
           "compute_clisis"]