"""
Runtime instrumentation for NicheCompass training.

The question this exists to answer is "which parts of a run get faster when I
add a GPU, and which do not". Answering it needs three things that a plain
´time.time()´ around a block does not give:

1. CUDA is asynchronous. A timer around GPU work measures kernel *launch*, not
   execution, and the cost silently lands on whatever forces the next
   synchronization. Probes that bracket GPU work therefore declare
   ´SYNC_CUDA´ and the profiler drains the stream at both boundaries.
2. Under ´DistributedDataParallel´ the ranks are not symmetric, and time spent
   blocked in a collective belongs to whichever rank arrived first. A probe
   that lumps waiting in with working makes a load imbalance look like work.
   ´WAIT´ probes synchronize and then barrier, so the wait is attributed to
   imbalance rather than to the collective.
3. Knowing a stage is slow is not the same as knowing it should have got
   faster. Every probe declares a ´kind´ saying how its cost is *expected* to
   behave as processes are added, and the report compares that expectation
   against what happened. That is what turns a table of numbers into an answer.

´time.perf_counter´ is used rather than CUDA events on purpose: the leading
suspects for fixed per-step cost here are CPU side (inline neighbor sampling,
device-to-host syncs, Python level reductions over device tensors), and CUDA
events cannot see any of them.

Nothing here changes numerics. With profiling off, every call goes to a
singleton whose methods return immediately, and ´iterate´ hands back the
original iterable rather than wrapping it.
"""

import json
import os
import warnings
from collections import namedtuple
from time import perf_counter
from typing import List, Optional, Union

import numpy as np
import torch

from .distributed import (barrier,
                          get_rank,
                          get_world_size,
                          is_initialized)

###############################################################################
## Probe declarations ##
###############################################################################

# How a stage's cost is expected to behave as processes are added. The report
# checks the measurement against this, which is what makes a scaling
# regression visible rather than merely present.
KIND_SHARDED = "sharded"      # should divide by world size
KIND_FIXED = "fixed/step"     # per step, but independent of batch size
KIND_REPLICATED = "replicated"  # same cost on every rank, whatever world size
KIND_COMM = "comm"            # collective; grows with world size
KIND_WAIT = "wait"            # blocked on other ranks; measures imbalance
KIND_MIXED = "mixed"          # genuinely both; no expectation asserted

SYNC_NONE = 0      # pure CPU region, ´perf_counter´ is already exact
SYNC_CUDA = 1      # brackets GPU work, drain the stream at both ends
SYNC_BARRIER = 2   # drain, then barrier, and call the barrier itself the cost

PROFILE_MODES = ("off", "phase", "step", "imbalance")
_LEVEL = {"off": 0, "phase": 1, "step": 2, "imbalance": 3}

Probe = namedtuple("Probe", "name kind level sync help")

# The single source of truth. Every rank builds its accumulator vector from
# this tuple in this order, so every rank reports the same keys in the same
# positions and the gather cannot misalign, even if a rank never entered a
# given probe.
PROBES = (
    # --- per training step (level 2) -------------------------------------
    Probe("loader.sample", KIND_SHARDED, 2, SYNC_NONE,
          "neighbor sampling and batch assembly, train and validation"),
    Probe("train.to_device", KIND_SHARDED, 2, SYNC_CUDA,
          "host to device copy of the batch"),
    Probe("train.forward", KIND_SHARDED, 2, SYNC_CUDA,
          "forward passes and loss (one fused call under DDP)"),
    Probe("train.loss_item", KIND_FIXED, 2, SYNC_NONE,
          "per key .item() on the loss dict; forces a device sync each"),
    Probe("train.backward", KIND_MIXED, 2, SYNC_CUDA,
          "backward, including the DDP gradient all reduce"),
    Probe("train.optimizer", KIND_FIXED, 2, SYNC_CUDA,
          "gradient clipping and the optimizer step"),
    # --- per validation step (level 2) -----------------------------------
    Probe("val.forward", KIND_SHARDED, 2, SYNC_CUDA,
          "validation forward passes"),
    Probe("val.accumulate", KIND_SHARDED, 2, SYNC_NONE,
          "accumulating validation predictions on the host"),
    # --- once per epoch (level 1) ----------------------------------------
    Probe("val.gather", KIND_COMM, 1, SYNC_NONE,
          "all gather of the validation shards back to full size"),
    Probe("val.metrics", KIND_REPLICATED, 1, SYNC_NONE,
          "eval_metrics on the FULL validation set, on every rank"),
    Probe("epoch.reduce", KIND_COMM, 1, SYNC_NONE,
          "all reduce of the scalar epoch logs"),
    Probe("epoch.early_stop", KIND_REPLICATED, 1, SYNC_NONE,
          "early stopping bookkeeping and the best model deepcopy"),
    # --- imbalance (level 3) ---------------------------------------------
    Probe("wait.epoch_end", KIND_WAIT, 3, SYNC_BARRIER,
          "time this rank waits for the slowest rank, per epoch"),
)

PROBE_INDEX = {p.name: i for i, p in enumerate(PROBES)}
PROBE_BY_NAME = {p.name: p for p in PROBES}

# Counters are not times; they give the report its denominators.
COUNTERS = ("n_train_steps", "n_val_steps", "n_val_entries")

LOG_PREFIX = "[nc-prof]"
JSON_PREFIX = "[nc-prof-json]"


def resolve_profile_mode(profile: Optional[Union[bool, str]]=None) -> str:
    """
    Work out the profiling mode from an explicit argument and the environment.

    Parameters
    ----------
    profile:
        ´True´ means ´"step"´, ´False´ and ´None´ fall through to the
        ´NICHECOMPASS_PROFILE´ environment variable and then to ´"off"´.

    Returns
    ----------
    mode:
        One of ´PROFILE_MODES´.
    """
    if profile is True:
        return "step"
    if isinstance(profile, str):
        if profile not in PROFILE_MODES:
            # An explicit argument is a programming error, so this is loud.
            raise ValueError(
                f"´profile´ is {profile!r}, which is not one of "
                f"{PROFILE_MODES}.")
        return profile
    env = os.environ.get("NICHECOMPASS_PROFILE")
    if env:
        if env not in PROFILE_MODES:
            # An environment variable is not, and a stale entry in a shell
            # profile must never fail a training run.
            warnings.warn(
                f"NICHECOMPASS_PROFILE is {env!r}, which is not one of "
                f"{PROFILE_MODES}. Profiling is disabled.")
            return "off"
        return env
    return "off"


###############################################################################
## The profiler ##
###############################################################################

class _NullSpan:
    """Context manager that does nothing, reused so that a disabled probe
    allocates nothing at all."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_SPAN = _NullSpan()


class NullProfiler:
    """
    The profiler used when profiling is off.

    Every method returns immediately, and ´iterate´ returns the iterable it
    was given rather than a generator, so a disabled run does not even pay for
    an extra frame per batch.
    """
    enabled = False
    mode = "off"
    level = 0

    def region(self, name):
        return _NULL_SPAN

    def iterate(self, iterable, name):
        return iterable

    def count(self, name, n=1):
        return None

    def wait(self, name):
        return None

    def note(self, key, value):
        return None

    def epoch_end(self, wall_s):
        return None

    def finalize(self, training_time_s=None):
        return None


NULL_PROFILER = NullProfiler()


class _Span:
    """
    Context manager for one probe.

    One instance is preallocated per probe name. Probe names are unique and
    never recursive, so no two spans of the same name are ever concurrently
    live and reuse is safe.
    """
    __slots__ = ("prof", "idx", "sync", "t0")

    def __init__(self, prof, idx, sync):
        self.prof = prof
        self.idx = idx
        self.sync = sync
        self.t0 = 0.0

    def __enter__(self):
        if self.sync == SYNC_CUDA and self.prof.cuda:
            # Drain whatever the previous stage queued, so its kernels are not
            # charged to this probe.
            torch.cuda.synchronize(self.prof.device)
        self.t0 = perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync == SYNC_CUDA and self.prof.cuda:
            torch.cuda.synchronize(self.prof.device)
        self.prof.acc[self.idx] += perf_counter() - self.t0
        self.prof.calls[self.idx] += 1
        # Never swallow an exception.
        return False


class RunProfiler:
    """
    Accumulates per stage timings for one process and reports them.

    Parameters
    ----------
    mode:
        One of ´PROFILE_MODES´. Probes above the corresponding level are
        inert.
    device:
        Device this process trains on, used for the CUDA synchronizations.
    output_dir:
        If given, each rank writes its own raw timings there as JSON.
    meta:
        Free-form provenance recorded in the report header. The thread counts
        and process topology belong here: a per rank CPU allocation that
        changes with the process count silently changes every CPU bound stage,
        and comparing two runs without it is comparing two machines.
    """
    enabled = True

    def __init__(self,
                 mode: str,
                 device: torch.device,
                 output_dir: Optional[str]=None,
                 meta: Optional[dict]=None):
        self.mode = mode
        self.level = _LEVEL[mode]
        self.device = device
        self.cuda = (device is not None and getattr(device, "type", None)
                     == "cuda" and torch.cuda.is_available())
        self.output_dir = output_dir
        self.meta = dict(meta) if meta else {}
        self.acc = np.zeros(len(PROBES), dtype=np.float64)
        self.calls = np.zeros(len(PROBES), dtype=np.int64)
        self.counters = {key: 0 for key in COUNTERS}
        self.epoch_wall: List[float] = []
        # Spans are preallocated, but only for probes this level enables; the
        # rest resolve to the no-op span.
        self._spans = {}
        for probe in PROBES:
            if probe.level <= self.level:
                self._spans[probe.name] = _Span(self,
                                                PROBE_INDEX[probe.name],
                                                probe.sync)

    def region(self, name: str):
        """Time a block: ´with profiler.region("train.forward"): ...´."""
        return self._spans.get(name, _NULL_SPAN)

    def iterate(self, iterable, name: str):
        """
        Time only the ´__next__´ of an iterable.

        This is the one probe that cannot be a ´with´ block. The cost being
        measured, neighbor sampling, happens inside the loader's ´__next__´,
        which is driven by ´zip´ in the training loop, so there is no block to
        wrap.
        """
        if name not in self._spans:
            return iterable
        idx = PROBE_INDEX[name]
        return self._iterate(iterable, idx)

    def _iterate(self, iterable, idx):
        iterator = iter(iterable)
        while True:
            t0 = perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                # Only StopIteration is caught, so the deliberate errors the
                # loaders raise on an empty split still propagate unchanged.
                return
            self.acc[idx] += perf_counter() - t0
            self.calls[idx] += 1
            yield item

    def count(self, name: str, n: int=1):
        """Record a denominator, for example the number of optimizer steps."""
        if name in self.counters:
            self.counters[name] += n

    def wait(self, name: str):
        """
        Measure how long this rank waits for the slowest rank.

        The leading synchronize is what makes the number mean anything:
        entering the barrier with this rank's own kernels still queued would
        charge this rank's work to "waiting for others", which is exactly
        backwards.
        """
        probe = PROBE_BY_NAME.get(name)
        if (probe is None or probe.level > self.level
                or not is_initialized()):
            return
        if self.cuda:
            torch.cuda.synchronize(self.device)
        t0 = perf_counter()
        barrier()
        self.acc[PROBE_INDEX[name]] += perf_counter() - t0
        self.calls[PROBE_INDEX[name]] += 1

    def note(self, key: str, value):
        """Record provenance for the report header."""
        self.meta[key] = value

    def epoch_end(self, wall_s: float):
        """Close out an epoch, recording its wall time."""
        self.epoch_wall.append(float(wall_s))

    # ---------------------------------------------------------------- report

    def _gather_per_rank(self) -> Optional[np.ndarray]:
        """
        Collect every rank's accumulator vector.

        A gather rather than a sum: the report needs the spread across ranks,
        because under DDP the slowest rank sets the pace and an average hides
        exactly the imbalance worth seeing.
        """
        if not is_initialized():
            return self.acc.reshape(1, -1)
        world_size = get_world_size()
        local = torch.as_tensor(self.acc, dtype=torch.float64,
                                device=self.device)
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, local)
        return torch.stack(gathered).cpu().numpy()

    def finalize(self, training_time_s: Optional[float]=None) -> Optional[str]:
        """
        Gather across ranks and render the report.

        Collective: every rank must call this, and it must be called at a
        point every rank reaches.

        Returns
        ----------
        report:
            The rendered report on the main process, ´None´ elsewhere.
        """
        try:
            per_rank = self._gather_per_rank()
        except Exception as exc:  # pragma: no cover - diagnostics only
            warnings.warn(f"Profiling failed to gather timings: {exc}. "
                          "The run itself is unaffected.")
            return None

        if self.output_dir is not None:
            self._write_json(per_rank)

        if is_initialized() and get_rank() != 0:
            return None
        return render_report(per_rank,
                             calls=self.calls,
                             counters=self.counters,
                             epoch_wall=self.epoch_wall,
                             meta=self.meta,
                             mode=self.mode,
                             training_time_s=training_time_s)

    def _write_json(self, per_rank: np.ndarray):
        rank = get_rank() if is_initialized() else 0
        path = os.path.join(self.output_dir, f"nc_profile_rank{rank}.json")
        payload = {"mode": self.mode,
                   "rank": rank,
                   "world_size": per_rank.shape[0],
                   "meta": self.meta,
                   "probes": [p.name for p in PROBES],
                   "kinds": [p.kind for p in PROBES],
                   "seconds": self.acc.tolist(),
                   "calls": self.calls.tolist(),
                   "counters": self.counters,
                   "epoch_wall_s": self.epoch_wall}
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:  # pragma: no cover - diagnostics only
            warnings.warn(f"Could not write the profile to {path}: {exc}.")


def make_profiler(mode: str,
                  device: torch.device,
                  output_dir: Optional[str]=None,
                  meta: Optional[dict]=None):
    """
    Build a profiler, or the no-op singleton when profiling is off.

    Parameters
    ----------
    mode:
        One of ´PROFILE_MODES´.

    Returns
    ----------
    profiler:
        A ´RunProfiler´, or ´NULL_PROFILER´.
    """
    if mode == "off":
        return NULL_PROFILER
    return RunProfiler(mode=mode,
                       device=device,
                       output_dir=output_dir,
                       meta=meta)


###############################################################################
## Rendering ##
###############################################################################

def _fmt(seconds: float) -> str:
    if seconds >= 100:
        return f"{seconds:8.0f}"
    return f"{seconds:8.2f}"


def render_report(per_rank: np.ndarray,
                  calls: np.ndarray,
                  counters: dict,
                  epoch_wall: List[float],
                  meta: dict,
                  mode: str,
                  training_time_s: Optional[float]=None) -> str:
    """
    Render the timing table.

    Every line carries ´LOG_PREFIX´ so the whole report can be pulled out of a
    large scheduler log with a single grep, and the same numbers are emitted
    once more as a JSON line for machine comparison of two runs.
    """
    world_size = per_rank.shape[0]
    # The slowest rank sets the pace, so that is the column that matters; the
    # spread next to it is the load imbalance.
    worst = per_rank.max(axis=0)
    mean = per_rank.mean(axis=0)
    total_worst = worst.sum()
    n_epochs = max(len(epoch_wall), 1)

    lines = []
    add = lines.append
    add(f"{LOG_PREFIX} " + "=" * 74)
    add(f"{LOG_PREFIX} RUNTIME PROFILE (mode={mode}, world_size={world_size}, "
        f"epochs={len(epoch_wall)})")
    for key in sorted(meta):
        add(f"{LOG_PREFIX}   {key}: {meta[key]}")
    for key in COUNTERS:
        if counters.get(key):
            add(f"{LOG_PREFIX}   {key}: {counters[key]}")
    add(f"{LOG_PREFIX} " + "-" * 74)
    add(f"{LOG_PREFIX} {'stage':<20} {'kind':<12} {'max/rank':>9} "
        f"{'%':>6} {'per epoch':>10} {'spread':>8}")
    add(f"{LOG_PREFIX} " + "-" * 74)

    order = np.argsort(-worst)
    for i in order:
        probe = PROBES[i]
        if calls[i] == 0 and worst[i] == 0:
            continue
        share = (100 * worst[i] / total_worst) if total_worst > 0 else 0.0
        spread = (worst[i] - per_rank[:, i].min()) if world_size > 1 else 0.0
        add(f"{LOG_PREFIX} {probe.name:<20} {probe.kind:<12} "
            f"{_fmt(worst[i])} {share:5.1f}% {_fmt(worst[i] / n_epochs)} "
            f"{spread:7.2f}s")

    add(f"{LOG_PREFIX} " + "-" * 74)
    add(f"{LOG_PREFIX} {'measured total':<20} {'':<12} {_fmt(total_worst)}")
    if training_time_s is not None:
        residual = training_time_s - total_worst
        add(f"{LOG_PREFIX} {'training_time':<20} {'':<12} "
            f"{_fmt(training_time_s)}")
        add(f"{LOG_PREFIX} {'unattributed':<20} {'':<12} {_fmt(residual)}"
            "   (uninstrumented, plus profiling overhead)")

    # The whole point of the exercise: say which stages could ever have got
    # faster, and which could not. But only when enough of the time is
    # actually accounted for. A confident ratio computed over a few percent of
    # the run is worse than no ratio at all, because it reads as an answer:
    # at mode='phase' the entire per step loop is uninstrumented by design,
    # and the surviving per epoch probes are nearly all replicated, so the
    # naive ratio says "1.01x, give up" about a run whose own stage budget
    # says 1.86x.
    # Only ´replicated´ genuinely cannot shrink. ´fixed/step´ is paid once
    # per optimizer step and is independent of the batch size, so under the
    # default ´per_process´ convention - where an epoch has ´world_size´ times
    # fewer steps - it shrinks with the process count like anything else.
    # Counting it here would have understated the ceiling on every default
    # run.
    # ´fixed/step´ shrinks only when the STEP COUNT shrinks, which happens
    # under ´per_process´ and not under ´global´, where the step count is
    # unchanged. Assuming per_process here overstated the ceiling on every
    # global run, which is exactly the run someone makes to compare against
    # one device.
    fixed_shrinks = meta.get("batch_size_scaling", "per_process") != "global"
    unshrinkable = ({KIND_REPLICATED} if fixed_shrinks
                    else {KIND_REPLICATED, KIND_FIXED})
    replicated = sum(worst[i] for i, p in enumerate(PROBES)
                     if p.kind in unshrinkable)
    coverage = (total_worst / training_time_s
                if training_time_s and training_time_s > 0 else 1.0)
    add(f"{LOG_PREFIX} " + "-" * 74)
    if training_time_s:
        add(f"{LOG_PREFIX} probe coverage: {100 * coverage:.1f}% of "
            "training time")
    disabled = sorted(p.name for p in PROBES if p.level > _LEVEL[mode])
    if disabled:
        add(f"{LOG_PREFIX} NOT instrumented at mode={mode!r}: "
            f"{', '.join(disabled)}")
        add(f"{LOG_PREFIX}   their cost sits in 'unattributed' above; rerun "
            "with --profile step to break it out")
    if coverage < 0.5:
        # Refuse the verdict rather than extrapolate from a sliver.
        add(f"{LOG_PREFIX} Too little of the run is attributed for a scaling "
            "verdict.")
        add(f"{LOG_PREFIX} Of what WAS measured, {replicated:.1f}s of "
            f"{total_worst:.1f}s does not shrink with more processes.")
    elif total_worst > 0:
        which = ("replicated stages only; 'fixed/step' shrinks under "
                 "per_process" if fixed_shrinks
                 else "replicated and fixed/step; the step count does not "
                      "shrink under global")
        add(f"{LOG_PREFIX} does NOT shrink with more processes "
            f"({which}): "
            f"{replicated:.1f}s of {total_worst:.1f}s "
            f"({100 * replicated / total_worst:.0f}%)")
        if replicated > 0:
            # Amdahl over the training region alone.
            par = total_worst - replicated
            for n in (2, 4):
                ceiling = total_worst / (replicated + par / n)
                add(f"{LOG_PREFIX}   => best possible training speed-up at "
                    f"{n} processes: {ceiling:.2f}x")
    add(f"{LOG_PREFIX} " + "=" * 74)

    payload = {"mode": mode,
               "world_size": world_size,
               "epochs": len(epoch_wall),
               "probes": {PROBES[i].name: {"max": float(worst[i]),
                                           "mean": float(mean[i]),
                                           "kind": PROBES[i].kind,
                                           "calls": int(calls[i])}
                          for i in range(len(PROBES))
                          if calls[i] or worst[i]},
               "counters": counters,
               "training_time_s": training_time_s,
               "meta": meta}
    lines.append(f"{JSON_PREFIX} {json.dumps(payload, separators=(',', ':'))}")
    return "\n".join(lines)


###############################################################################
## Script level stage budget ##
###############################################################################

class StageBudget:
    """
    Coarse whole-run timing for the stages outside ´Trainer.train´.

    Deliberately separate from ´RunProfiler´: this runs in the entry script,
    where there is no process group to gather over and where the stages that
    matter are the ones that are either duplicated on every rank or run on the
    main process alone while the others idle. Those are invisible from inside
    the training loop, and they were 74% of wall clock when last measured.
    """

    def __init__(self, enabled: bool=True):
        self.enabled = enabled
        self.stages = []
        self._t_start = perf_counter()
        self._t_mark = self._t_start

    def stage(self, name: str, parallel: str="all ranks"):
        return _StageSpan(self, name, parallel)

    def checkpoint(self, name: str, parallel: str="all ranks"):
        """
        Close the stage that has been running since the last checkpoint.

        A one line alternative to ´stage´ for instrumenting a long script,
        where wrapping each region in a ´with´ block would mean reindenting
        most of the file for no benefit.
        """
        now = perf_counter()
        if self.enabled:
            self.stages.append((name, parallel, now - self._t_mark))
        self._t_mark = now

    def report(self) -> str:
        total = perf_counter() - self._t_start
        lines = [f"{LOG_PREFIX} " + "=" * 74,
                 f"{LOG_PREFIX} WHOLE RUN STAGE BUDGET",
                 f"{LOG_PREFIX} " + "-" * 74,
                 f"{LOG_PREFIX} {'stage':<26} {'runs on':<14} "
                 f"{'seconds':>9} {'%':>6}"]
        lines.append(f"{LOG_PREFIX} " + "-" * 74)
        accounted = 0.0
        for name, parallel, seconds in self.stages:
            accounted += seconds
            share = 100 * seconds / total if total > 0 else 0.0
            lines.append(f"{LOG_PREFIX} {name:<26} {parallel:<14} "
                         f"{_fmt(seconds)} {share:5.1f}%")
        lines.append(f"{LOG_PREFIX} " + "-" * 74)
        lines.append(f"{LOG_PREFIX} {'total wall clock':<26} {'':<14} "
                     f"{_fmt(total)}")
        lines.append(f"{LOG_PREFIX} {'unaccounted':<26} {'':<14} "
                     f"{_fmt(total - accounted)}")

        training = sum(s for n, _, s in self.stages if n == "training")
        if training > 0 and total > 0:
            serial = total - training
            lines.append(f"{LOG_PREFIX} " + "-" * 74)
            lines.append(f"{LOG_PREFIX} training is {100 * training / total:.0f}%"
                         " of wall clock; the rest cannot be sped up by adding"
                         " GPUs")
            for n in (2, 4):
                lines.append(f"{LOG_PREFIX}   => whole run ceiling at {n} GPUs,"
                             f" even with perfect training scaling: "
                             f"{total / (serial + training / n):.2f}x")
        lines.append(f"{LOG_PREFIX} " + "=" * 74)
        payload = {"total_wall_s": total,
                   "stages": [{"name": n, "runs_on": p, "seconds": s}
                              for n, p, s in self.stages]}
        lines.append(f"{JSON_PREFIX} "
                     f"{json.dumps(payload, separators=(',', ':'))}")
        return "\n".join(lines)


class _StageSpan:
    __slots__ = ("budget", "name", "parallel", "t0")

    def __init__(self, budget, name, parallel):
        self.budget = budget
        self.name = name
        self.parallel = parallel
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, *exc):
        if self.budget.enabled:
            self.budget.stages.append(
                (self.name, self.parallel, perf_counter() - self.t0))
        return False
