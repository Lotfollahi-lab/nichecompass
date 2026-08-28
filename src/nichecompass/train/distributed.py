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
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


def is_distributed_launch() -> bool:
    """
    Indicate whether the current process was launched as part of a distributed
    job, by checking the environment variables that ´torchrun´ sets.

    This is deliberately a check of the environment rather than of
    ´torch.distributed.is_initialized´, so that it can be called before the
    process group exists.
    """
    return ("RANK" in os.environ
            and "WORLD_SIZE" in os.environ
            and int(os.environ.get("WORLD_SIZE", "1")) > 1)


def is_initialized() -> bool:
    """Indicate whether a process group has been initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the rank of the current process, or ´0´ if not distributed."""
    return dist.get_rank() if is_initialized() else 0


def get_world_size() -> int:
    """Return the number of processes, or ´1´ if not distributed."""
    return dist.get_world_size() if is_initialized() else 1


def get_local_rank() -> int:
    """
    Return the rank of the current process within its node, which is the index
    of the device it should use. Falls back to ´0´ if not distributed.
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return get_rank()


def is_main_process() -> bool:
    """
    Indicate whether the current process is the one that should print, log,
    write files and mutate ´adata´.
    """
    return get_rank() == 0


def init_distributed(backend: Optional[str]=None) -> bool:
    """
    Initialize the process group from the environment that ´torchrun´ provides,
    and bind this process to its device.

    Parameters
    ----------
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
    if not is_distributed_launch():
        return False
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
    dist.init_process_group(backend=backend)
    return True


def cleanup_distributed():
    """Destroy the process group if one exists."""
    if is_initialized():
        barrier()
        dist.destroy_process_group()


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


def broadcast_object(obj, source_rank: int=0):
    """
    Broadcast a picklable python object from ´source_rank´ to every process,
    and return it. Returns the object unchanged if not distributed.

    Used so that decisions taken on one process, such as whether early stopping
    has triggered, are taken by every process identically.
    """
    if not is_initialized():
        return obj
    object_list = [obj if get_rank() == source_rank else None]
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
