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

import ast
import collections
import io
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from nichecompass.train.distributed import (_tensors_to_cpu,
                                            all_gather_numpy,
                                            all_reduce_mean,
                                            detect_launcher,
                                            get_local_rank,
                                            get_rank,
                                            get_world_size,
                                            init_distributed,
                                            is_distributed_launch,
                                            is_main_process,
                                            shard_indices,
                                            unwrap_model)

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_distributed_worker.py")


def run_workers(world_size: int, n_obs: int, out_dir: str,
                port: int, extra_args: list=None) -> list:
    """
    Launch ´world_size´ worker processes with the environment that ´torchrun´
    sets, and return what each of them wrote.
    """
    extra_args = extra_args or []
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
             "--n_obs", str(n_obs), "--out_dir", out_dir] + extra_args,
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
## Which launcher started the process ##
###############################################################################

LAUNCHER_ENVIRONMENTS = [
    ("torchrun", {"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "2"}),
    ("Open MPI", {"OMPI_COMM_WORLD_RANK": "2", "OMPI_COMM_WORLD_SIZE": "4",
                  "OMPI_COMM_WORLD_LOCAL_RANK": "2"}),
    ("MPICH", {"PMI_RANK": "2", "PMI_SIZE": "4", "MPI_LOCALRANKID": "2"}),
    ("PMIx", {"PMIX_RANK": "2", "PMIX_SIZE": "4", "PMIX_LOCAL_RANK": "2"}),
]
LAUNCHER_VARIABLES = [variable for _, environment in LAUNCHER_ENVIRONMENTS
                      for variable in environment]


@pytest.fixture
def no_launcher(monkeypatch):
    """Remove every launcher variable, so each test starts from a clean slate."""
    for variable in LAUNCHER_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


@pytest.mark.parametrize("name,environment", LAUNCHER_ENVIRONMENTS)
def test_every_supported_launcher_is_detected(no_launcher, name, environment):
    # An LSF site commonly starts one process per GPU with mpirun rather than
    # torchrun, so the MPI variables have to be understood too
    for variable, value in environment.items():
        no_launcher.setenv(variable, value)
    launcher = detect_launcher()
    assert launcher is not None
    assert launcher == (name, 2, 4, 2)
    assert is_distributed_launch() is True
    assert get_local_rank() == 2


@pytest.mark.parametrize("name,environment", LAUNCHER_ENVIRONMENTS)
def test_one_process_is_not_a_distributed_launch(no_launcher, name,
                                                 environment):
    # One process is the single device path, whichever launcher nominally
    # started it
    for variable, value in environment.items():
        no_launcher.setenv(variable, "0" if "RANK" in variable
                           or "LOCALRANKID" in variable else "1")
    assert detect_launcher() is None
    assert is_distributed_launch() is False


def test_no_launcher_is_not_a_distributed_launch(no_launcher):
    assert detect_launcher() is None
    assert is_distributed_launch() is False


def test_torchrun_wins_over_a_stale_mpi_environment(no_launcher):
    # Both sets present should resolve to torchrun, which is the explicit one
    no_launcher.setenv("RANK", "1")
    no_launcher.setenv("WORLD_SIZE", "2")
    no_launcher.setenv("LOCAL_RANK", "1")
    no_launcher.setenv("OMPI_COMM_WORLD_RANK", "3")
    no_launcher.setenv("OMPI_COMM_WORLD_SIZE", "8")
    assert detect_launcher()[0] == "torchrun"
    assert detect_launcher()[1:] == (1, 2, 1)


def test_a_missing_local_rank_falls_back_to_the_global_rank(no_launcher):
    # Some MPI builds do not export a local rank; on one node it equals the
    # global rank
    no_launcher.setenv("OMPI_COMM_WORLD_RANK", "3")
    no_launcher.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert detect_launcher() == ("Open MPI", 3, 4, 3)


def test_an_mpi_launch_without_a_rendezvous_raises_explaining_itself(
        no_launcher):
    # torchrun sets MASTER_ADDR and MASTER_PORT; mpirun does not, so the
    # submission script has to, and the failure has to say so
    no_launcher.setenv("OMPI_COMM_WORLD_RANK", "0")
    no_launcher.setenv("OMPI_COMM_WORLD_SIZE", "2")
    no_launcher.delenv("MASTER_ADDR", raising=False)
    no_launcher.delenv("MASTER_PORT", raising=False)
    with pytest.raises(RuntimeError, match="MASTER_ADDR and MASTER_PORT"):
        init_distributed(backend="gloo")


def test_only_one_process_is_main_before_the_group_exists(no_launcher):
    """
    The guards that keep experiment tracking and writing results to one process
    run BEFORE the process group is created and AFTER it is destroyed. Asking
    torch.distributed at those moments answers 0 on every process, so every
    process would believe it is the main one. This is the bug that made four
    ranks race to create the same MLflow tables.
    """
    no_launcher.setenv("OMPI_COMM_WORLD_SIZE", "4")
    mains = []
    for rank in range(4):
        no_launcher.setenv("OMPI_COMM_WORLD_RANK", str(rank))
        assert get_rank() == rank
        assert get_world_size() == 4
        mains.append(is_main_process())
    assert mains == [True, False, False, False], (
        "exactly one process may be the main one")


def test_rank_is_zero_and_world_size_one_without_any_launcher(no_launcher):
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True


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


@pytest.mark.parametrize("world_size,port", [(2, 29527), (4, 29528)])
def test_gradients_match_while_a_loss_term_is_still_warming_up(
        world_size, port, tmp_path):
    """
    NicheCompass reports the edge reconstruction loss in ´global_loss´ from the
    first epoch but leaves it out of ´optim_loss´ for the first
    ´n_epochs_no_edge_recon´ epochs, and does the same with the contrastive
    loss. Those epochs therefore run a different set of parameters through the
    backward pass than through the forward, and the gradients still have to
    match a single process. The worker runs two iterations, since a reducer
    left in a bad state after one only shows it on the next forward.
    """
    sys.path.insert(0, os.path.dirname(WORKER_PATH))
    from _distributed_worker import single_process_gradients

    n_obs = 8 * world_size
    expected = single_process_gradients(n_obs, edge_recon_active=False)
    results = run_workers(world_size, n_obs, str(tmp_path), port,
                          extra_args=["--no_edge_recon"])

    for result in results:
        for name, expected_gradient in expected.items():
            torch.testing.assert_close(
                result["gradients"][name], expected_gradient,
                rtol=1e-5, atol=1e-6,
                msg=f"gradient of {name} diverged on rank {result['rank']}")
    # The graph decoder is untouched while the edge loss is off, so its
    # gradient is what a reducer waiting on it would have been waiting for
    assert torch.count_nonzero(expected["source_theta"]) == 0


def test_the_wrapper_returns_only_the_optimized_loss_with_its_graph():
    """
    Reads ´_JointForwardModule.forward´ rather than running it. Everything the
    wrapper returns is walked by ´find_unused_parameters´, and the documented
    contract is that every returned tensor derived from a parameter takes part
    in the backward pass, so every entry other than ´optim_loss´ leaves
    detached. Asserted on the source because torch 2.x turned out to tolerate
    the breach at runtime, which makes a behavioural test silently vacuous.
    """
    trainer_source = io.open(
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))),
            "src", "nichecompass", "train", "trainer.py"),
        encoding="utf-8").read()
    tree = ast.parse(trainer_source)
    wrapper = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef)
                   and node.name == "_JointForwardModule")
    forward = next(node for node in wrapper.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "forward")
    body = ast.unparse(forward)
    # The loss is computed here, not by the caller, because it uses parameters
    assert ".loss(" in body
    assert "detach()" in body
    assert "'optim_loss'" in body or '"optim_loss"' in body


###############################################################################
## What crosses the process boundary ##
###############################################################################

TRAINER_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src", "nichecompass", "train", "trainer.py")


def _trainer_function(name: str) -> ast.FunctionDef:
    tree = ast.parse(io.open(TRAINER_PATH, encoding="utf-8").read())
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def test_tensors_are_moved_to_host_and_the_structure_is_kept():
    Pair = collections.namedtuple("Pair", ["left", "right"])
    original = {"weights": collections.OrderedDict(a=torch.ones(2)),
                "list": [torch.zeros(1), 3, "text"],
                "pair": Pair(torch.ones(1), None),
                "scalar": 7,
                "none": None}
    mapped = _tensors_to_cpu(original)

    assert isinstance(mapped["weights"], collections.OrderedDict)
    assert isinstance(mapped["pair"], Pair) and mapped["pair"].right is None
    assert isinstance(mapped["list"], list)
    assert mapped["list"][1] == 3 and mapped["list"][2] == "text"
    assert mapped["scalar"] == 7 and mapped["none"] is None
    for tensor in (mapped["weights"]["a"], mapped["list"][0],
                   mapped["pair"].left):
        assert tensor.device.type == "cpu"
    torch.testing.assert_close(mapped["weights"]["a"], torch.ones(2))


def test_tensors_to_cpu_detaches_so_a_graph_is_never_pickled():
    # Pickling a tensor that still carries a graph raises, and a state dict
    # taken from a model mid-training can hold one
    parameter = torch.nn.Parameter(torch.ones(2))
    mapped = _tensors_to_cpu({"grad": parameter * 2})
    assert mapped["grad"].requires_grad is False


def test_broadcast_object_maps_to_host_before_it_pickles():
    """
    Asserted on the source rather than by running it, because the failure needs
    two GPUs in exclusive process mode: a pickled tensor records its device,
    and every receiver restores it onto the SENDER's device, which it is not
    allowed to open.
    """
    source = io.open(os.path.join(os.path.dirname(TRAINER_PATH),
                                 "distributed.py"), encoding="utf-8").read()
    function = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "broadcast_object")
    # Walked as a tree rather than matched as text, so that the explanation in
    # the docstring cannot satisfy the test on its own
    mapped_at, collective_at = None, None
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if getattr(node.func, "id", "") == "_tensors_to_cpu":
                assert [ast.unparse(argument) for argument in node.args] == \
                    ["obj"]
                mapped_at = node.lineno if mapped_at is None else mapped_at
            if getattr(node.func, "attr", "") == "broadcast_object_list":
                collective_at = node.lineno
    assert mapped_at is not None, "the object is pickled with its devices"
    assert collective_at is not None
    assert mapped_at < collective_at, "mapped to host after the broadcast"


@pytest.mark.parametrize("world_size,port", [(2, 29529), (4, 29530)])
def test_a_state_dictionary_broadcast_arrives_in_host_memory(
        world_size, port, tmp_path):
    results = run_workers(world_size, 8 * world_size, str(tmp_path), port)
    for result in results:
        # Including the sender, so that every process then moves the weights to
        # its own device by the same route
        assert result["received_devices"] == ["cpu"], (
            f"rank {result['rank']} received tensors on "
            f"{result['received_devices']}")
        assert result["received_keys"] == results[0]["received_keys"]
        torch.testing.assert_close(result["received_sum"],
                                   results[0]["received_sum"])


###############################################################################
## Releasing the process group ##
###############################################################################

def _cleanup_with(monkeypatch, barrier_fails: bool, destroy_fails: bool):
    from nichecompass.train import distributed as module

    def explode(*args, **kwargs):
        raise RuntimeError("Cuda failure 'CUDA-capable device(s) is/are busy'")
    monkeypatch.setattr(module, "is_initialized", lambda: True)
    monkeypatch.setattr(module, "barrier",
                        explode if barrier_fails else lambda *a, **k: None)
    monkeypatch.setattr(module.dist, "destroy_process_group",
                        explode if destroy_fails else lambda *a, **k: None)
    return module


def test_a_healthy_run_whose_release_fails_is_told_it_is_unaffected(
        monkeypatch):
    """
    The synchronization inside ´cleanup_distributed´ doubles as a health check:
    if every process completed a collective there, the collectives the run
    depended on completed too, and a later failure is confined to handing the
    communicator back. This is the case the farm hits on three of four ranks.
    """
    module = _cleanup_with(monkeypatch, barrier_fails=False, destroy_fails=True)
    with pytest.warns(UserWarning) as recorded:
        module.cleanup_distributed()
    messages = " ".join(str(warning.message) for warning in recorded)
    assert "nothing this run produced is affected" in messages
    assert "unverified" not in messages


def test_an_unhealthy_communicator_is_not_reported_as_benign(monkeypatch):
    """
    A failure of the synchronization itself is a different situation: the
    communicator was already unhealthy, so the gradient reduction and the
    gathered metrics cannot be assumed to have completed. Reporting that as
    the benign teardown failure would hide a run whose results are wrong.
    """
    module = _cleanup_with(monkeypatch, barrier_fails=True, destroy_fails=True)
    with pytest.warns(UserWarning) as recorded:
        module.cleanup_distributed()
    messages = " ".join(str(warning.message) for warning in recorded)
    assert "cannot be assumed to have completed" in messages
    assert "unverified" in messages
    assert "nothing this run produced is affected" not in messages


@pytest.mark.parametrize("failing", ["barrier", "destroy_process_group"])
def test_a_failed_teardown_is_reported_and_not_raised(monkeypatch, failing):
    """
    On a cluster whose GPUs are allocated in an exclusive compute mode, NCCL's
    teardown cannot touch the peer devices it opened on the other processes,
    and three of four ranks died with

        ncclUnhandledCudaError: Call to CUDA function failed.
        Cuda failure 'CUDA-capable device(s) is/are busy or unavailable'

    after a run that had already trained, reloaded its best state and computed
    every metric. Losing all of that because the communicator could not be
    handed back is the wrong trade.
    """
    module = _cleanup_with(monkeypatch,
                           barrier_fails=(failing == "barrier"),
                           destroy_fails=(failing == "destroy_process_group"))
    with pytest.warns(UserWarning, match="process group"):
        module.cleanup_distributed()


def test_the_group_is_released_exactly_once_per_process():
    # ´cleanup_distributed´ synchronizes before it releases, so a barrier at
    # the call site as well made every process wait twice for nothing
    source = io.open(os.path.join(os.path.dirname(os.path.dirname(
        TRAINER_PATH)), "models", "nichecompass.py"), encoding="utf-8").read()
    tree = ast.parse(source)
    train = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "train")
    calls = [ast.unparse(node.func) for node in ast.walk(train)
             if isinstance(node, ast.Call)]
    assert calls.count("barrier") == 0, (
        "cleanup_distributed already synchronizes, so a barrier beside it "
        "makes every process wait twice")
    assert calls.count("cleanup_distributed") == 2, (
        "both the main and the non-main path have to release the group")


def test_loading_onto_a_gpu_does_not_move_a_single_process_run():
    """
    ´load(use_cuda=True)´ used to be ´model.model.cuda()´, which loads onto
    whichever device is CURRENT. Binding it to the local rank fixes the case
    the feature is for -- four processes must not all load onto device 0 -- but
    naming device 0 unconditionally would silently relocate a single process
    load that a caller had aimed elsewhere with ´torch.cuda.set_device´.
    """
    source = io.open(os.path.join(os.path.dirname(os.path.dirname(
        TRAINER_PATH)), "models", "basemodelmixin.py"),
        encoding="utf-8").read()
    call = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "cuda")
    argument = ast.unparse(call.args[0])
    assert "get_local_rank()" in argument
    assert "is_distributed_launch()" in argument, (
        "the device is named unconditionally, so a single process load no "
        f"longer honours the current device: {argument}")
    # None is what restores Module.cuda()'s own default
    assert "None" in argument


###############################################################################
## Failure modes with no symptom ##
###############################################################################

def test_cycling_an_empty_loader_raises_instead_of_spinning():
    """
    The node loaders are consumed through ´_cycle_iterable´, which rebuilds its
    iterator whenever it runs out. On an EMPTY loader that is an infinite loop
    in pure python: no collective is entered, so no watchdog fires, and a
    distributed run hangs with no output and no error until its wall clock
    expires. ´drop_last´ is set for distributed runs, so a split shorter than
    one batch does produce an empty loader.
    """
    # Compiled out of the source rather than imported, so that the test does
    # not pull in the plotting dependencies the rest of that module has
    utils_path = os.path.join(os.path.dirname(TRAINER_PATH), "utils.py")
    tree = ast.parse(io.open(utils_path, encoding="utf-8").read())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_cycle_iterable")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 utils_path, "exec"), namespace)
    _cycle_iterable = namespace["_cycle_iterable"]

    with pytest.raises(ValueError, match="no batches"):
        next(_cycle_iterable([]))

    # and it still cycles, which is the whole point of it
    cycled = _cycle_iterable([1, 2])
    assert [next(cycled) for _ in range(5)] == [1, 2, 1, 2, 1]


def test_an_empty_loader_is_rejected_before_the_first_epoch():
    # The message at setup is worth more than the hang, and it also catches an
    # empty EDGE loader, which does not hang but silently gives an epoch of no
    # iterations and NaN logs
    # In Trainer.__init__, which is before any epoch and before any collective
    tree = ast.parse(io.open(TRAINER_PATH, encoding="utf-8").read())
    trainer = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    setup = next(node for node in trainer.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "__init__")
    assert "yields no batches" in ast.unparse(setup)


def test_the_process_group_is_given_an_explicit_timeout():
    """
    Torch's default for ´nccl´ is ten minutes. At the end of training the other
    processes wait in a barrier while the main process computes the latent
    representation over the whole dataset, and on an atlas that single pass can
    exceed ten minutes -- which would abort the waiting processes after training
    had already succeeded.
    """
    source = io.open(os.path.join(os.path.dirname(TRAINER_PATH),
                                 "distributed.py"), encoding="utf-8").read()
    function = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "init_distributed")
    call = next(node for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "init_process_group")
    assert "timeout" in [keyword.arg for keyword in call.keywords], (
        "init_process_group is called without a timeout, so the ten minute "
        "nccl default applies")


###############################################################################
## Side effects that every process needs ##
###############################################################################

def test_early_stopping_is_not_called_on_the_main_process_only():
    """
    ´is_early_stopping´ reduces the learning rate on the optimizer and records
    the best model state, both of which are per process. Calling it on the main
    process alone left the others on the original learning rate as soon as the
    scheduler fired, so they applied different updates to the same averaged
    gradients from that step on.
    """
    train = _trainer_function("train")
    for node in ast.walk(train):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "is_early_stopping"):
            continue
        # Not the ´x if is_main_process() else None´ shape that caused it
        for parent in ast.walk(train):
            if isinstance(parent, ast.IfExp) and any(
                    call is node for call in ast.walk(parent)):
                pytest.fail("is_early_stopping is guarded by a conditional "
                            f"expression: {ast.unparse(parent)}")
    assert any(isinstance(node, ast.Call)
               and getattr(node.func, "attr", "") == "is_early_stopping"
               for node in ast.walk(train)), "the call disappeared"


def test_no_main_process_guard_mutates_the_learning_rate():
    """
    The general shape of the bug above: a per process side effect inside a
    block that only one process runs. The learning rate is the one that
    actually bit, so it is the one guarded here.
    """
    tree = ast.parse(io.open(TRAINER_PATH, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not any(isinstance(call, ast.Call)
                   and getattr(call.func, "id", "") == "is_main_process"
                   for call in ast.walk(node.test)):
            continue
        for statement in ast.walk(ast.Module(body=node.body,
                                             type_ignores=[])):
            # A write to param_group["lr"], not merely a mention of it: the
            # message that reports the new learning rate reads it, and only
            # one process should print that
            targets = (statement.targets
                       if isinstance(statement, ast.Assign)
                       else [statement.target]
                       if isinstance(statement, ast.AugAssign) else [])
            for target in targets:
                assert not (isinstance(target, ast.Subscript)
                            and getattr(target.value, "id", "")
                            == "param_group"), (
                    "a block only the main process runs changes the learning "
                    f"rate: {ast.unparse(statement)}")


def test_the_best_model_state_is_not_sent_over_the_interconnect():
    # Every process records it from the epoch every process agreed on, so
    # broadcasting it only risked the device tagging failure and a full copy
    # of the weights on the wire
    train = _trainer_function("train")
    for node in ast.walk(train):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "broadcast_object"):
            argument = ast.unparse(node.args[0])
            assert "state_dict" not in argument, argument


def test_the_per_epoch_metrics_are_gathered_before_they_are_computed():
    """
    ´eval_epoch´ appends the AUROC and friends straight to ´epoch_logs´, which
    does NOT go through the all-reduce the iteration level logs get. Computed
    per shard they would differ between processes and would not be comparable
    to a single device run, and ´early_stopping_metric´ may name one of them.
    """
    body = ast.unparse(_trainer_function("eval_epoch"))
    assert "all_gather_numpy" in body
    assert body.index("all_gather_numpy") < body.index("eval_metrics(")


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


###############################################################################
## Every mask has to travel with the model ##
###############################################################################

def test_every_decoder_mask_is_a_buffer_so_module_to_moves_it():
    """
    The decoder masks are combined with each other in the forward pass: the
    static mask is concatenated with the add-on mask, and the result is
    multiplied by the dynamic mask. If any of them is a plain attribute rather
    than a registered buffer, ´Module.to´ leaves that one on the CPU while
    moving the others, and the forward pass fails with a device mismatch as
    soon as the model is moved to a GPU.

    The check walks the syntax tree rather than matching text, because an
    earlier version of this test matched only ´self.x = ...´ assignments and
    was therefore blind to the add-on masks, which are registered under a
    computed name. Those masks then broke a four GPU run at the concatenation.
    Every mechanism that can bind an attribute is covered here.

    ´gene_peaks_mask_´ is deliberately exempt: its trailing underscore marks it
    as one of the public attributes pickled when a model is saved, so it stays
    a plain attribute and each of its uses aligns its device explicitly.
    """
    import ast
    from pathlib import Path

    # Located by path rather than imported, so the check stays free of the
    # module's heavy dependencies
    module_path = (Path(__file__).resolve().parent.parent
                   / "src" / "nichecompass" / "modules" / "vgpgae.py")
    assert module_path.is_file(), f"cannot find {module_path}"
    tree = ast.parse(module_path.read_text())

    exempt = {"gene_peaks_mask_"}
    registered = {ast.unparse(node.args[0]).strip("'\"")
                  for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and getattr(node.func, "attr", "") == "register_buffer"
                  and node.args}
    offenders = []
    for node in ast.walk(tree):
        # self.<name> = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and "mask" in target.attr.lower()
                        and getattr(target.value, "id", "") == "self"
                        and target.attr not in exempt
                        and target.attr not in registered):
                    offenders.append(f"line {node.lineno}: self.{target.attr}")
        # setattr(self, "<name>", ...), including a computed name
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "setattr"
                and len(node.args) >= 2
                and getattr(node.args[0], "id", "") == "self"):
            name = ast.unparse(node.args[1])
            if "mask" in name.lower():
                offenders.append(f"line {node.lineno}: setattr {name}")

    assert not offenders, (
        "these masks are bound as plain attributes, so Module.to will not "
        "move them and the forward pass will break on a GPU: "
        + "; ".join(offenders))
