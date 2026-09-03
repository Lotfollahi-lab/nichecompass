"""
This module contains helpers for distributed (multi-GPU) training of a
NicheCompass model.

Training is data parallel: every process holds a full copy of the model and of
the spatial graph, and each process works on a disjoint subset of the training
edges and nodes. Gradients are averaged across processes by
´DistributedDataParallel´, so that one optimizer step sees an effective batch of
´world_size´ times the per process batch size.

Every function here is safe to call when training on a single device, in which
case it either returns the single process answer or does nothing. Nothing in
this module is imported at model construction time, so a single device run does
not pay for it.
"""

import os
import warnings
from collections import OrderedDict
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


# Environment variables through which each supported launcher reports the rank
# of a process, the total number of processes and the rank within the node.
# ´torchrun´ is listed first, so an explicit torchrun environment always wins.
# The MPI entries matter because on an LSF cluster the established way to start
# one process per GPU is usually ´mpirun´ under the scheduler, not ´torchrun´.
_LAUNCHER_ENVIRONMENTS = (
    ("torchrun", "RANK", "WORLD_SIZE", "LOCAL_RANK"),
    ("Open MPI", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE",
     "OMPI_COMM_WORLD_LOCAL_RANK"),
    ("MPICH", "PMI_RANK", "PMI_SIZE", "MPI_LOCALRANKID"),
    ("PMIx", "PMIX_RANK", "PMIX_SIZE", "PMIX_LOCAL_RANK"),
)


def detect_launcher() -> Optional[tuple]:
    """
    Return the launcher that started this process, as a tuple of its name and
    the rank, world size and local rank it reported, or ´None´ if the process
    was not started by a recognized launcher with more than one process.

    Both ´torchrun´ and an MPI launcher are supported, because which one is
    used is a property of the cluster rather than of NicheCompass: ´torchrun´
    is the PyTorch default, while LSF sites commonly start one process per GPU
    with ´mpirun´ under the scheduler.
    """
    for name, rank_var, size_var, local_rank_var in _LAUNCHER_ENVIRONMENTS:
        if rank_var not in os.environ or size_var not in os.environ:
            continue
        world_size = int(os.environ[size_var])
        if world_size <= 1:
            # One process is the single device path, whichever launcher
            # nominally started it
            continue
        rank = int(os.environ[rank_var])
        local_rank = int(os.environ.get(local_rank_var, rank))
        return name, rank, world_size, local_rank
    return None


def is_distributed_launch() -> bool:
    """
    Indicate whether the current process was launched as part of a distributed
    job with more than one process.

    This is deliberately a check of the environment rather than of
    ´torch.distributed.is_initialized´, so that it can be called before the
    process group exists.
    """
    return detect_launcher() is not None


def is_initialized() -> bool:
    """Indicate whether a process group has been initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """
    Return the rank of the current process, or ´0´ if not distributed.

    Falls back to the rank the launcher reported when no process group exists,
    which is what makes this usable BEFORE the group is created and AFTER it is
    destroyed. Asking ´torch.distributed´ alone would answer ´0´ on every
    process at those moments, so every process would believe it is the main one
    and would duplicate the work that only one of them should do.
    """
    if is_initialized():
        return dist.get_rank()
    launcher = detect_launcher()
    return launcher[1] if launcher is not None else 0


def get_world_size() -> int:
    """
    Return the number of processes, or ´1´ if not distributed.

    Falls back to the launcher for the same reason as ´get_rank´.
    """
    if is_initialized():
        return dist.get_world_size()
    launcher = detect_launcher()
    return launcher[2] if launcher is not None else 1


def get_local_rank() -> int:
    """
    Return the rank of the current process within its node, which is the index
    of the device it should use. Falls back to ´0´ if not distributed.

    The local rank comes from whichever launcher started the process, since it
    is the only reliable way to know which of the node's devices belongs to
    this process. Deriving it from the global rank would be wrong on any run
    that spans more than one node.
    """
    launcher = detect_launcher()
    if launcher is not None:
        return launcher[3]
    return get_rank()


def is_main_process() -> bool:
    """
    Indicate whether the current process is the one that should print, log,
    write files and mutate ´adata´.

    Correct at any point in the run, including before the process group is
    created and after it is destroyed, because it goes through ´get_rank´.
    That matters: the guards around experiment tracking run before training
    starts, and the guards around writing results run after it finishes.
    """
    return get_rank() == 0


# How long a process may sit in one collective before the watchdog aborts it.
# Torch's own default for ´nccl´ is ten minutes, which is not enough here: at
# the end of training the other processes wait in a barrier while the main
# process computes the latent representation over the WHOLE dataset, and on an
# atlas that single pass can take longer than ten minutes. Overrun there would
# abort the waiting processes and take the run down after training had already
# succeeded. An hour is far longer than any legitimate collective and still
# short enough to surface a genuine deadlock in one job's wall clock.
DEFAULT_COLLECTIVE_TIMEOUT_MINUTES = 60


def init_distributed(backend: Optional[str]=None,
                     timeout_minutes: Optional[float]=None) -> bool:
    """
    Initialize the process group from the environment that the launcher
    provides, and bind this process to its device.

    Works with ´torchrun´ and with an MPI launcher such as ´mpirun´, since
    which of the two starts one process per GPU is a property of the cluster.
    An MPI launcher reports the rank under its own variable names and does not
    set the rendezvous address, so the submission script has to export
    ´MASTER_ADDR´ and ´MASTER_PORT´ and forward them to the ranks.

    Parameters
    ----------
    timeout_minutes:
        How long a process may wait in a single collective before the watchdog
        aborts it. Defaults to ´DEFAULT_COLLECTIVE_TIMEOUT_MINUTES´, or to the
        environment variable ´NICHECOMPASS_COLLECTIVE_TIMEOUT_MINUTES´ when
        that is set.
    backend:
        Communication backend. Defaults to ´nccl´ when CUDA is available, which
        is the only backend that is fast for GPU tensors, and to ´gloo´
        otherwise, which is what the CPU tests use.

    Returns
    ----------
    initialized:
        ´True´ if this process is part of an initialized process group.
    """
    if is_initialized():
        return True
    launcher = detect_launcher()
    if launcher is None:
        return False
    launcher_name, rank, world_size, local_rank = launcher

    # ´torch.distributed´ reads the rendezvous from RANK, WORLD_SIZE,
    # MASTER_ADDR and MASTER_PORT. ´torchrun´ sets all four; an MPI launcher
    # sets none of them under those names, so they are filled in from what it
    # did report. The submit script is responsible for exporting MASTER_ADDR
    # and MASTER_PORT and for forwarding them to the ranks, since only it knows
    # the allocation.
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        raise RuntimeError(
            f"Started by {launcher_name} with {world_size} processes, but "
            "MASTER_ADDR and MASTER_PORT are not set, so the processes cannot "
            "find each other. Export both in the submission script and "
            "forward them to the ranks, which for ´mpirun´ means "
            "´-x MASTER_ADDR -x MASTER_PORT´. ´torchrun´ sets them itself.")

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    # The device is bound BEFORE the process group is created. NCCL binds its
    # communicator to whichever device is current, so a process that joined the
    # group before selecting its device would create its context on device 0
    # along with every other process. Under an exclusive compute mode that is
    # an abort rather than merely wasted memory, and it presents as the GPU
    # mode being at fault when it is not.
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    if timeout_minutes is None:
        timeout_minutes = float(os.environ.get(
            "NICHECOMPASS_COLLECTIVE_TIMEOUT_MINUTES",
            DEFAULT_COLLECTIVE_TIMEOUT_MINUTES))
    dist.init_process_group(backend=backend,
                            timeout=timedelta(minutes=timeout_minutes))
    return True


def cleanup_distributed():
    """
    Release the process group, if one exists.

    Every step is tolerant of failure, and that is deliberate. By the time this
    runs the model is trained, the best state is loaded and every metric is
    computed, so failing to hand the communicator back is not a reason to fail
    the run. And it does fail on a cluster that allocates its GPUs in an
    exclusive compute mode: NCCL's teardown releases the peer resources it
    opened on the OTHER processes' devices, and ´cudaSetDevice´ on a device
    another process holds exclusively is refused, so three of four ranks died
    with

        NCCL WARN Cuda failure 'CUDA-capable device(s) is/are busy or
        unavailable'
        ncclUnhandledCudaError: Call to CUDA function failed.

    on device 0, which belongs to the one rank that got through. Reported as a
    warning and then left alone: the processes are about to exit and the driver
    reclaims everything they held.
    """
    if not is_initialized():
        return
    try:
        barrier()
    except Exception as error:
        warnings.warn(f"Synchronizing the processes before releasing the "
                      f"process group failed: {error}. Training itself had "
                      f"already finished, so this is reported rather than "
                      f"raised.")
    try:
        dist.destroy_process_group()
    except Exception as error:
        warnings.warn(f"Releasing the process group failed: {error}. This "
                      f"happens on clusters whose GPUs are allocated in an "
                      f"exclusive compute mode, where NCCL cannot touch the "
                      f"peer devices it needs to release. Everything the run "
                      f"produced is already complete, and the driver reclaims "
                      f"what the processes held when they exit, so this is "
                      f"reported rather than raised.")


def barrier():
    """
    Synchronize all processes, and do nothing if not distributed.

    The device is named explicitly, because a bare barrier lets NCCL pick one
    and every process would pick device 0, which is an abort under an exclusive
    compute mode.
    """
    if not is_initialized():
        return
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[get_local_rank()])
    else:
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Replace a tensor by its mean across all processes, in place, and return it.

    Used for quantities that every process has to agree on exactly, such as the
    running mean absolute gene program scores that drive gene program pruning.
    Without this the processes would prune different gene programs and their
    models would diverge into different architectures.
    """
    if not is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def all_reduce_sum_scalar(value: float, device: torch.device) -> float:
    """Return the sum of a python scalar across all processes."""
    if not is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def all_gather_numpy(array: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Concatenate a one dimensional numpy array across all processes.

    Used to reconstruct the validation predictions and labels that the
    evaluation metrics are computed from, which each process accumulates only
    for the part of the validation set it was given. Arrays may differ in
    length between processes, so the lengths are exchanged first.
    """
    if not is_initialized():
        return array
    tensor = torch.as_tensor(np.ascontiguousarray(array.astype(np.float64)),
                             device=device)
    world_size = get_world_size()
    local_length = torch.tensor([tensor.numel()], dtype=torch.int64,
                                device=device)
    lengths = [torch.zeros(1, dtype=torch.int64, device=device)
               for _ in range(world_size)]
    dist.all_gather(lengths, local_length)
    lengths = [int(length.item()) for length in lengths]
    max_length = max(lengths)
    # ´all_gather´ needs equally sized buffers, so the tensors are padded and
    # the padding is removed again afterwards
    padded = torch.zeros(max_length, dtype=torch.float64, device=device)
    padded[:tensor.numel()] = tensor
    gathered = [torch.zeros(max_length, dtype=torch.float64, device=device)
                for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return np.concatenate([gathered[i][:lengths[i]].cpu().numpy()
                           for i in range(world_size)])


def _tensors_to_cpu(obj):
    """
    Copy every tensor in ´obj´ to host memory, leaving the structure around
    them intact. Recurses through dicts, lists and tuples; anything else is
    returned unchanged.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, OrderedDict):
        return OrderedDict((key, _tensors_to_cpu(value))
                           for key, value in obj.items())
    if isinstance(obj, dict):
        return {key: _tensors_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_tensors_to_cpu(item) for item in obj]
    if isinstance(obj, tuple):
        items = tuple(_tensors_to_cpu(item) for item in obj)
        # namedtuples take their fields positionally, not as one iterable
        return type(obj)(*items) if hasattr(obj, "_fields") else items
    return obj


def broadcast_object(obj, source_rank: int=0):
    """
    Broadcast a picklable python object from ´source_rank´ to every process,
    and return it. Returns the object unchanged if not distributed.

    Used so that decisions taken on one process, such as whether early stopping
    has triggered, are taken by every process identically.

    Tensors are moved to host memory before they are broadcast. This is not an
    optimization: ´broadcast_object_list´ works by pickling, and pickling a
    tensor records the device it is on. Every receiving process would then
    restore it onto the SENDER's device, and under the exclusive process
    compute mode that GPUs are usually allocated with on a cluster, only the
    sender may open a context there. The others fail with

        CUDA error: CUDA-capable device(s) is/are busy or unavailable

    which is what a state dictionary broadcast from the main process did on
    four H100s. Host tensors carry no device, so every process restores them
    locally and can move them wherever it needs them.
    """
    if not is_initialized():
        return obj
    object_list = [_tensors_to_cpu(obj) if get_rank() == source_rank else None]
    dist.broadcast_object_list(object_list, src=source_rank)
    return object_list[0]


def shard_indices(indices: torch.Tensor,
                  rank: Optional[int]=None,
                  world_size: Optional[int]=None,
                  drop_remainder: bool=True) -> torch.Tensor:
    """
    Split a one dimensional index tensor into a disjoint, contiguous shard per
    process.

    Every process is given exactly the same number of indices, so that every
    process runs the same number of optimizer steps. That matters because the
    processes synchronize on every step: a process that ran out of batches
    early would leave the others waiting forever on the next gradient
    reduction, which hangs rather than raising. The remainder that does not
    divide evenly is dropped, which discards at most ´world_size - 1´ items per
    epoch.

    The shard is strided rather than contiguous, so that each process sees a
    spread of the dataset rather than one contiguous block of it. For spatial
    data a contiguous block of node indices is often a contiguous region of the
    tissue, which would give each process a biased view of the batch.

    Parameters
    ----------
    indices:
        Indices to split, for example the indices of the training nodes or the
        columns of the training edge label index.
    rank:
        Process rank. Defaults to the rank of the current process.
    world_size:
        Number of processes. Defaults to the size of the current process group.
    drop_remainder:
        If ´True´, truncate to a length divisible by ´world_size´ so that every
        shard has exactly the same size.

    Returns
    ----------
    shard:
        The indices assigned to this process. Identical to ´indices´ when there
        is only one process, so that the single device path is unchanged.
    """
    if rank is None:
        rank = get_rank()
    if world_size is None:
        world_size = get_world_size()
    if world_size == 1:
        return indices
    n_indices = indices.shape[0]
    n_per_rank = n_indices // world_size
    if n_per_rank == 0:
        raise ValueError(
            f"Cannot split {n_indices} items across {world_size} processes, "
            "since that would leave at least one process without data. Use "
            "fewer processes or a larger dataset.")
    if drop_remainder:
        # Truncating before striding is what makes every shard exactly equal
        indices = indices[:n_per_rank * world_size]
    return indices[rank::world_size]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying model, unwrapping ´DistributedDataParallel´ if the
    model is wrapped.

    Saving, the post training methods and every attribute lookup on the model
    have to go through this, since the wrapper puts a ´module.´ prefix in front
    of every parameter name and does not forward arbitrary attributes.
    """
    return getattr(model, "module", model)
