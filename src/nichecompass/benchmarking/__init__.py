# This is a trick to make jax use the right cudnn version (needs to be executed
# before importing scanpy)
#import jax.numpy as jnp
#temp_array = jnp.array([1, 2, 3])
#temp_idx = jnp.array([1])
#temp_array[temp_idx]

from .cas import compute_avg_cas, compute_cas
from .clisis import compute_clisis
from .gcs import compute_avg_gcs, compute_gcs
from .metrics import compute_benchmarking_metrics
from .mlami import compute_mlami
from .nasw import compute_nasw

__all__ = ["compute_avg_cas",
           "compute_avg_gcs",
           "compute_benchmarking_metrics",
           "compute_cas",
           "compute_clisis",
           "compute_gcs",
           "compute_mlami",
           "compute_nasw"]