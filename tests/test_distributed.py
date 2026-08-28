"""
Tests for the distributed (multi-GPU) training support.

The machine that runs these tests does not need a GPU. The correctness of the
distributed path is a property of how the work is split and of which quantities
are reduced, not of the device it runs on, so the tests launch real processes
over the ´gloo´ backend on CPU. That exercises the same ´init_distributed´
entry point, the same collectives and the same ´DistributedDataParallel´
gradient reduction that a multi-GPU run uses.

The central test is ´test_gradients_match_a_single_process´, which is the claim
the feature rests on: training with several processes over a global batch
produces the same gradients as training with one process over that batch.

What these tests do NOT cover is anything specific to CUDA: device binding by
local rank, the ´nccl´ backend, and the actual speedup. Those need a machine
with several GPUs.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from nichecompass.train.distributed import (all_gather_numpy,
                                            all_reduce_mean,
                                            get_rank,
                                            get_world_size,
                                            is_distributed_launch,
                                            is_main_process,
                                            shard_indices,
                                            unwrap_model)

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_distributed_worker.py")


def run_workers(world_size: int, n_obs: int, out_dir: str,
                port: int) -> list:
    """
    Launch ´world_size´ worker processes with the environment that ´torchrun´
    sets, and return what each of them wrote.
    """
    processes = []
    for rank in range(world_size):
        env = dict(os.environ,
                   RANK=str(rank),
                   LOCAL_RANK=str(rank),
                   WORLD_SIZE=str(world_size),
                   MASTER_ADDR="127.0.0.1",
                   MASTER_PORT=str(port),
                   OMP_NUM_THREADS="1")
        processes.append(subprocess.Popen(
            [sys.executable, WORKER_PATH,
             "--n_obs", str(n_obs), "--out_dir", out_dir],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    results = []
    for rank, process in enumerate(processes):
        stdout, stderr = process.communicate(timeout=300)
        if process.returncode != 0:
            pytest.fail(f"worker {rank} failed with code "
                        f"{process.returncode}:\n{stderr.decode()}")
        results.append(torch.load(os.path.join(out_dir, f"rank_{rank}.pt"),
                                  weights_only=False))
    return results


###############################################################################
## Splitting the work across processes ##
###############################################################################

@pytest.mark.parametrize("n_indices,world_size", [
    (100, 2), (100, 4), (101, 2), (7, 3), (1000, 8),
])
def test_shards_are_equally_sized_and_disjoint(n_indices, world_size):
    indices = torch.arange(n_indices)
    shards = [shard_indices(indices, rank=rank, world_size=world_size)
              for rank in range(world_size)]
    # Equal size, because a process that ran out of batches early would leave
    # the others waiting forever on the next gradient reduction
    assert len({len(shard) for shard in shards}) == 1
    assert len(shards[0]) == n_indices // world_size
    # Disjoint, so that no observation is seen twice per epoch
    seen = torch.cat(shards)
    assert len(seen) == len(torch.unique(seen))
    assert set(seen.tolist()).issubset(set(indices.tolist()))


def test_shards_drop_at_most_world_size_minus_one_items():
    shards = [shard_indices(torch.arange(101), rank=rank, world_size=4)
              for rank in range(4)]
    assert 101 - sum(len(shard) for shard in shards) < 4


def test_a_single_process_gets_everything_unchanged():
    # The single device path has to be untouched, so the shard is the input
    indices = torch.arange(37)
    assert torch.equal(shard_indices(indices, rank=0, world_size=1), indices)


def test_shards_are_strided_rather_than_contiguous():
    # A contiguous block of spatial node indices is often a contiguous region
    # of the tissue, which would give each process a biased view of the batch
    shard = shard_indices(torch.arange(100), rank=0, world_size=4)
    assert shard[0].item() == 0 and shard[1].item() == 4


def test_more_processes_than_data_raises():
    with pytest.raises(ValueError, match="without data"):
        shard_indices(torch.arange(3), rank=0, world_size=8)


###############################################################################
## Behaviour without a process group ##
###############################################################################

def test_helpers_are_inert_without_a_process_group():
    # Everything has to be safe to call on the single device path
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True
    tensor = torch.tensor([1., 2., 3.])
    assert torch.equal(all_reduce_mean(tensor.clone()), tensor)
    array = np.array([1., 2.])
    assert np.array_equal(all_gather_numpy(array, torch.device("cpu")), array)


def test_a_plain_run_is_not_a_distributed_launch(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert is_distributed_launch() is False


def test_a_world_size_of_one_is_not_a_distributed_launch(monkeypatch):
    # Launching with one process is the single device path, not a distributed
    # one, so no process group is created for it
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert is_distributed_launch() is False


def test_unwrap_model_passes_through_an_unwrapped_model():
    model = torch.nn.Linear(2, 2)
    assert unwrap_model(model) is model


###############################################################################
## The claim the feature rests on ##
###############################################################################

@pytest.mark.parametrize("world_size,port", [(2, 29517), (4, 29518)])
def test_gradients_match_a_single_process(world_size, port, tmp_path):
    """
    Training with several processes over a global batch has to produce the same
    gradients as training with one process over that batch. This holds because
    every NicheCompass loss term is a mean over the batch and
    ´DistributedDataParallel´ averages gradients across processes.
    """
    sys.path.insert(0, os.path.dirname(WORKER_PATH))
    from _distributed_worker import single_process_gradients

    n_obs = 8 * world_size
    expected = single_process_gradients(n_obs)
    results = run_workers(world_size, n_obs, str(tmp_path), port)

    for result in results:
        for name, expected_gradient in expected.items():
            torch.testing.assert_close(
                result["gradients"][name], expected_gradient,
                rtol=1e-5, atol=1e-6,
                msg=f"gradient of {name} diverged on rank {result['rank']}")


def test_every_process_ends_up_with_the_same_pruning_quantity(tmp_path):
    # The gene program pruning decision is irreversible, so processes that
    # disagreed here would permanently train different models
    results = run_workers(world_size=3, n_obs=24, out_dir=str(tmp_path),
                          port=29519)
    for result in results:
        torch.testing.assert_close(result["reduced"], torch.full((3,), 2.))


def test_validation_predictions_are_gathered_across_processes(tmp_path):
    results = run_workers(world_size=3, n_obs=24, out_dir=str(tmp_path),
                          port=29520)
    for result in results:
        gathered = result["gathered"].numpy()
        # The processes contribute 2, 3 and 4 values respectively
        assert len(gathered) == 2 + 3 + 4
        assert sorted(np.unique(gathered).tolist()) == [0., 1., 2.]


def test_the_processes_cover_the_batch_exactly_once(tmp_path):
    results = run_workers(world_size=4, n_obs=32, out_dir=str(tmp_path),
                          port=29521)
    seen = torch.cat([result["shard"] for result in results])
    assert len(seen) == 32
    assert sorted(seen.tolist()) == list(range(32))


def test_rank_helpers_report_the_process_group(tmp_path):
    results = run_workers(world_size=2, n_obs=16, out_dir=str(tmp_path),
                          port=29522)
    assert [result["rank"] for result in results] == [0, 1]
    assert all(result["world_size"] == 2 for result in results)
    assert [result["is_main"] for result in results] == [True, False]
    assert all(result["unwraps"] for result in results)
