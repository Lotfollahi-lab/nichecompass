# Multi-GPU training

NicheCompass can split training across several GPUs. This page explains what the option does, what it
guarantees numerically, how to launch it, and what it does not cover.

Training on a single device is completely unaffected. The option is off by default and, when it is off, the
code takes exactly the same path it took before multi-GPU support existed.

## 1. How to use it

Multi-GPU training uses one process per GPU, so the script has to be launched as a distributed job:

```bash
torchrun --nproc_per_node=4 train_nichecompass_reference_model.py --multi_gpu <other arguments>
```

In your own code, pass `multi_gpu=True` to `train`:

```python
model = NicheCompass(adata, ...)
model.train(n_epochs=400,
            edge_batch_size=512,
            multi_gpu=True)
```

`torchrun` runs the whole script once per GPU. `multi_gpu=True` without such a launch raises immediately
with an explanatory error, rather than silently training on one device.

### On an HPC cluster

Ready-to-edit submitters are in the reproducibility repository at
`analysis/data_analysis/submit_lsf_sanger.sh` and `analysis/data_analysis/submit_slurm.sh`.

**LSF.** Which launcher starts the per-GPU processes is a property of the cluster, not of NicheCompass.
`torchrun` is the PyTorch default, but LSF sites commonly start one rank per GPU with `mpirun` under the
scheduler instead, and `init_distributed` supports both: it reads the rank from whichever launcher's
environment variables are present (`RANK`/`WORLD_SIZE`/`LOCAL_RANK` for `torchrun`,
`OMPI_COMM_WORLD_*`, `PMI_*` or `PMIX_*` for an MPI launcher).

An MPI launcher does not set the rendezvous address, so the submitting script has to export
`MASTER_ADDR` and `MASTER_PORT` and forward them to the ranks — for `mpirun` that means
`-x MASTER_ADDR -x MASTER_PORT`. If they are missing, `init_distributed` raises and says so rather than
hanging.

The submitter for the Sanger farm is `analysis/data_analysis/submit_lsf_sanger.sh`:

```bash
bash submit_lsf_sanger.sh --n_epochs 1 --n_epochs_all_gps 0   # 4 GPU smoke run
N_GPUS=1 bash submit_lsf_sanger.sh --n_epochs 1               # single device baseline
DRY_RUN=1 bash submit_lsf_sanger.sh                           # print the job, submit nothing
```

It is a submitter rather than a job script, because `#BSUB` directives are read before any shell runs and
cannot reference variables: building the job body from variables is the only way to keep the `-gpu num=`
request and the launcher's process count from drifting apart. It emits, for four GPUs:

```bash
#BSUB -q training-parallel
#BSUB -G team361                     # REQUIRED, see below
#BSUB -U lotfollahi-training-parallel
#BSUB -n 24
#BSUB -gpu "num=4:gmem=80000:mode=exclusive_process:block=yes"
#BSUB -M 200G
#BSUB -R "select[mem>200G] rusage[mem=200G] span[ptile=24]"
```

Three of those are site requirements that are easy to miss:

- **`-G <group>` is mandatory.** Without it the esub rejects the job before it is queued, with
  *"Sorry no available user group specified for this job"*. `bugroup -w | grep -w "$USER"` lists the groups
  you belong to.
- **The `training-parallel` queue is used together with an advance reservation** (`-U`). `brsvs` lists the
  reservations available to you.
- **`unset LSB_AFFINITY_HOSTFILE`** before `mpirun`, and load the scheduler-aware OpenMPI module
  (`ISG/experimental/fg12/openmpi/...-lsf`). NCCL also needs `NCCL_NVLS_ENABLE=0` on NVSwitch nodes whose
  driver rejects the multicast setup.

Run `probe_lsf_gpu_allocation.sh` once before a long run. It submits the same resource request and reports
whether the queue starts the job once or once per slot, how `num=` was interpreted, the memory limit with
its unit, and whether the ranks can bind to distinct GPUs and all-reduce over NCCL.

**Slurm** (`DATA_DIR=/path/to/h5ads bash submit_slurm.sh --n_epochs 100`). The submitter generates the
`#SBATCH` directives and runs the committed `_slurm_job_body.sh`, exactly as the LSF one does. `DATA_DIR`
is required and is the folder holding `{dataset}_{batch}.h5ad`, so the data can live anywhere the compute
nodes can see rather than inside the repository. `SLURM_PARTITION`, `N_GPUS`, `N_NODES`, `MEM_GB`, `WALL`
and `SLURM_ACCOUNT` are all settable from the environment.

*Asking for a particular GPU model needs care.* Clusters label GPU models in one of two ways and they are
not interchangeable — a typed gres (`--gres=gpu:a100:4`) or a node feature (`--constraint=a100`). Find out
which applies before submitting:

```bash
sinfo -o '%20P %10G %40f'
```

`%G` shows the gres — `gpu:a100:4` rather than a bare `gpu:4` means types are defined — and `%f` shows the
features. Set `GPU_GRES` or `GPU_CONSTRAINT` to match. This matters because **a type request the scheduler
does not understand is not an error**: it is silently satisfied by whatever was free, and the first sign
would be timings that do not match the hardware you thought you had. The job body therefore asserts the
model it actually received against `REQUIRE_GPU_MODEL` and fails in seconds if it is wrong.

If you edit the submitter, note that the directives are joined with empty entries *removed*. A blank line
is not a comment, and Slurm stops reading `#SBATCH` at the first line that is not one, so an unset optional
directive left as an empty string would silently discard every directive below it — including the GPU
request.

The generated directives look like this:

```bash
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
```

Request **one task per node**, not one per GPU. `torchrun` starts the per-GPU processes itself; asking
Slurm for four tasks as well would start four copies of `torchrun` and hence sixteen processes.

**Slurm, several nodes** (`sbatch --nodes=2 submit_slurm.sh`). The nodes have to find each other, so a
rendezvous endpoint on the first node is needed:

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

srun --kill-on-bad-exit=1 torchrun \
    --nnodes="$SLURM_NNODES" --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train_nichecompass_reference_model.py --multi_gpu <other arguments>
```

Deriving the port from the job id keeps two jobs that share a node from colliding. Multi-node runs are
only worth it once a single node's GPUs are saturated: the gradient reduction then crosses the network on
every step.

Multi-node under LSF works the same way as the single-node case, since `mpirun` already reads the
allocation from the scheduler: raise the node count and set `MASTER_ADDR` to the first host of
`$LSB_MCPU_HOSTS`, which the submitter does. It is only worth it once a single node's GPUs are saturated,
because the gradient reduction then crosses the network on every step.

### Three things that will bite you

- **`--nproc_per_node=1` with `--multi_gpu` raises.** One process is the single-device path, so ask for
  `--multi_gpu` only when you are actually requesting several GPUs.
- **Host memory scales with the number of processes**, not GPU memory. See section 6.
- **Populate the gene program caches first.** See section 7.

**Notebooks.** A notebook is a single process, so it cannot drive several GPUs this way. Run the
single-device path in notebooks, or move training into a script and launch it with `torchrun`. This is a
real limitation and not an oversight: the alternative, spawning worker processes from inside `train`, would
have to send a copy of `adata` to every worker, which is slower than the training it accelerates for the
dataset sizes where multi-GPU is worth having.

## 2. What is guaranteed

**The batch sizes are per process by default.** `batch_size_scaling` decides how `edge_batch_size` and
`node_batch_size` are read when `multi_gpu=True`:

| | `"per_process"` (default) | `"global"` |
|---|---|---|
| the given batch is | each process's | the whole run's |
| effective batch | × `world_size` | unchanged |
| optimizer steps per epoch | ÷ `world_size` | unchanged |
| comparable to a single-GPU run | no | yes |
| where the speedup comes from | there are fewer steps | each step is cheaper |

`"per_process"` is the default because that is where the speedup is, and the measurement is unambiguous. A
step on the reference configuration costs about 39 ms, of which about 24 ms — **62%** — does not scale with
the batch at all: the gradient all-reduce, the synchronous neighbour sampling, the two per-iteration
`.item()` synchronizations. Under `"global"` the step *count* never falls, so that fixed 62% is paid 2,256
times per epoch whether you have one device or eight, which caps the whole thing. Measured 1 GPU against
2 GPUs, `"global"` returns **1.23×**; extrapolated to four devices it is about 1.4×, against about **4×**
for `"per_process"`.

Two consequences follow from the larger effective batch, and neither is handled for you:

- **The learning rate is not scaled.** A `world_size`-fold larger batch usually wants the learning rate
  scaled with it, linearly or by its square root. `reduce_lr_on_plateau` absorbs some of this but is not a
  substitute.
- **A `"per_process"` run is not a control for a single-GPU run.** The two differ by construction, so this
  convention cannot be used to test whether the distributed implementation is correct.

`"global"` is what to use when the comparison matters more than the speed: reproducing a single-GPU result,
or testing the distributed path. It gives the same effective batch, the same number of optimizer steps and
the same learning rate as one device.

Neither setting has any effect on a single-device run. Every batch-size computation that depends on the
convention sits inside a `self.distributed_` guard, and a test asserts it, because every published result
came from one device.

The post-training latent pass is single-process and always uses the batch size the caller gave, never the
effective global one.

One thing this convention does *not* decide, contrary to an earlier version of this document: gene program
pruning. Pruning is driven by an exponential moving average with momentum 0.1, which is 99.5% converged
after 50 updates. Under `"per_process"` on four devices an epoch still contains 564 steps, so by
`n_epochs_all_gps` the average has had roughly 14,000 updates against a 50-update time constant — four
times fewer than under `"global"`, and equally converged. Both conventions prune from a fully converged
statistic.

**Gradients match a single-device run.** Every NicheCompass loss term is a mean over the batch — the
negative binomial reconstruction losses, the Kullback-Leibler term and the edge reconstruction
cross-entropy — or a function of the parameters alone, as the L1 and group lasso regularizers are.
`DistributedDataParallel` averages gradients across processes, so the average of the per-process means is
the mean over the global batch. `tests/test_distributed.py` asserts this against a single-process run with
two and with four real processes.

**What is not bit-identical.** Results will not reproduce a single-device run exactly, for two reasons that
are inherent rather than incidental. The processes draw different negative edges, which is deliberate:
sharing one random seed would make every process sample the same negative edges, so the extra devices
would recompute the same negatives instead of covering more of them. And the number of seed edges is
truncated to a multiple of the number of processes, which drops fewer than `world_size` edges per epoch.

> **If you are reproducing published results, train on a single GPU.** The multi-GPU path is for
> accelerating new runs. It targets the same objective and the same effective batch, but it is a different
> draw.

## 3. How the work is split

The graph is **never partitioned**. Both decoders reconstruct one-hop neighborhood-aggregated counts, so a
node's sampled subgraph has to contain its real neighbors; splitting the graph itself would silently change
what the source decoder is asked to reconstruct. Instead every process holds the full graph and is given a
disjoint subset of the **seeds**:

| Loader | Argument | Per process |
| :-- | :-- | :-- |
| `NeighborLoader` | `input_nodes` | a strided subset of the training node indices |
| `LinkNeighborLoader` | `edge_label_index` | a strided subset of the seed edge columns |

The subsets are strided rather than contiguous. A contiguous block of spatial node indices is often a
contiguous region of the tissue, which would give each process a systematically biased view of every batch.

Each process gets exactly the same number of seeds. That is a correctness requirement, not tidiness: the
processes synchronize on every optimizer step, so a process that ran out of batches first would leave the
others waiting forever on the next gradient reduction. That failure mode is a hang, not an error, which is
why the shards are equalized and `drop_last` is set in the distributed path only.

## 4. What is synchronized, and why

Data parallel training is only correct if the processes agree about everything that is not a gradient.
Four things had to be made collective.

**The gene program pruning statistic.** `running_mean_abs_mu` is a buffer updated by hand under
`torch.no_grad()`, which `DistributedDataParallel` does not touch. The sum of absolute gene program scores
and the number of nodes are now reduced separately across processes before the moving average is updated,
so the average is over the whole global batch. Without this the processes would compute different values,
derive different active gene program masks, and — because pruning is irreversible — permanently train
*different architectures* while reporting a single result.

**The dynamic decoder masks.** These were plain tensor attributes rather than registered buffers, so
`Module.to` did not move them to the GPU and no distributed machinery would have kept them consistent. They
are now non-persistent buffers, which fixes the device move and keeps the saved state dict unchanged, so
checkpoints written before this remain loadable.

**Early stopping and the best model.** The epoch-level losses are averaged across processes, the stopping
decision is taken on the main process and broadcast, and the best model state is broadcast before it is
loaded. If one process stopped while the others continued, the others would hang on the next reduction.

**The validation metrics.** Each process only evaluates its own shard, so the predictions and labels are
concatenated across processes before AUROC, AUPRC and the MSE scores are computed. Otherwise every process
would report a metric over a `world_size`-th of the validation set.

## 5. Nine structural details

**Two forward passes, one backward.** A training step runs the model twice, once for the node-level omics
decoder and once for the edge-level graph decoder, and then backpropagates a single combined loss.
`DistributedDataParallel` prepares its gradient reduction at the end of every forward pass and expects one
forward per backward, so wrapping the model directly would leave the first pass's gradients unreduced. The
two passes are therefore joined into a single forward by a small wrapper module. The two passes are
independent given the same parameters, so what is computed is unchanged.

**The loss is computed inside that wrapper, not by the trainer.** This is not a matter of tidiness. The
NicheCompass loss uses parameters *directly* rather than only the outputs of the forward pass: the negative
binomial dispersions `target_rna_theta` and `source_rna_theta`, and the decoder weights that the L1 and
group lasso regularizers penalize. A parameter used outside the wrapped forward has its gradient produced by
an autograd node the reducer never saw, so its hook fires a second time and the backward pass dies with

```
RuntimeError: Expected to mark a variable ready only once. ...
Parameter at index 1 with name model.source_rna_theta has been marked as ready twice.
```

The wrapper therefore returns the loss dictionary rather than the two model outputs. Only `optim_loss` keeps
its autograd graph; every other entry is detached on the way out, because `find_unused_parameters` walks the
graph of everything the forward returns and `global_loss` deliberately carries terms that `optim_loss` omits
while they warm up. The trainer only calls `.item()` on those entries, so nothing observable changes.

On a single device the loss is still called directly by the trainer, exactly as before.

**Nothing carrying a device crosses the process boundary.** `broadcast_object` moves every tensor it is
given into host memory before it broadcasts. This is not an optimization. `broadcast_object_list` works by
pickling, and pickling a tensor records the device it was on, so every receiving process restores it onto the
*sender's* device — which, under the `mode=exclusive_process` that GPUs are normally allocated with on a
cluster, only the sender may open. A state dictionary broadcast from the main process therefore died on three
of four H100s with

```
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

For the same reason the best model state is no longer broadcast at all: every process records it itself, at
the epoch every process agreed was the best, so `DistributedDataParallel` has already kept the two copies
identical and sending a full set of weights over the interconnect bought nothing.

**Everything a rank guard hides has to be free of per-process side effects.** `is_early_stopping` runs on
*every* process, not just the main one, because two of the three things it does are per-process: it reduces
the learning rate on that process's optimizer, and it records that process's best model state. Running it on
the main process alone left the others on the original learning rate as soon as the scheduler fired, and from
that step on they applied different updates to the same averaged gradients — a silent divergence that no
1-epoch smoke test can reach. It is safe to run everywhere because it reads only `epoch_logs`, which is
all-reduced. The decision is still broadcast, so agreement is guaranteed rather than inferred.

`eval_epoch` gathers the validation predictions across processes before computing AUROC and friends, for the
same reason `eval_end` does — and additionally because those entries go straight into `epoch_logs` without
passing through the all-reduce the iteration-level logs get. Computed per shard they would differ between
processes, would not be comparable to a single-GPU run, and `early_stopping_metric` may name one of them.

**Releasing the process group is allowed to fail, and the known failure is not a warning.**
`cleanup_distributed` synchronizes and then releases the group, and both steps are wrapped: a failure is
reported, never raised. The failure below is expected on every multi-GPU run on such a cluster, its
mechanism is understood, and it affects nothing — so it is reported as one line of the run's narrative on
**stdout**, not as a warning on stderr. Repeating a six-line warning on `world_size - 1` processes made a
healthy run look like a failing one, and stderr is the stream people scan when something has actually gone
wrong. Any release failure that does *not* match that signature is still a warning, so a future NCCL
rewording fails towards noise rather than silence. On a cluster whose GPUs are
allocated in an exclusive compute mode, NCCL's teardown releases the peer resources it opened on the *other*
processes' devices, and `cudaSetDevice` on a device another process holds exclusively is refused:

```
NCCL WARN Cuda failure 'CUDA-capable device(s) is/are busy or unavailable'
ncclUnhandledCudaError: Call to CUDA function failed.
```

That killed three of four ranks after a run had trained, reloaded its best state and computed every metric.

A run with `NCCL_DEBUG=INFO` established that this is what it is, rather than leaving it as the plausible
explanation it started as. NCCL prefixes every log line with the CUDA device the emitting thread has
current. On a two-process run the failing line comes from rank 1's process on a thread whose current device
is **0** — rank 0's — while that same process logs device 1 on every other line, so the prefix is not a
default. The peer transport is in use beforehand (`Channel NN/0 : 1[1] -> 0[0] via P2P/CUMEM` on all 24
channels), so what is being released is the peer mappings. Rank 0, with no peer above it to be refused by,
reports `Destroy COMPLETE`; the others fall back to `Abort COMPLETE`. Both finish, and neither touches
anything the run produced.

The reason a warning is right here is stronger than "nothing follows the teardown", and worth stating
correctly: on the main process `cleanup_distributed()` is the **last statement of `train()`**, and every
output — the neighbour graph, the UMAP, `adata.write`, `model.save` — happens *afterwards*, in the caller.
So on the process that matters, the teardown sits *before* 100% of the run's on-disk output. Letting it raise
would kill that process with everything still in memory, which is exactly what happened once already.

The two ways it can fail are not treated alike. The synchronisation comes first and doubles as a health
check: if every process completed a collective there, the collectives the run depended on completed too, and
a later failure is confined to handing the communicator back. If the *synchronisation* is what failed, the
communicator was already unhealthy, the gradient reduction and gathered metrics cannot be assumed to have
completed, and the warning says to treat the results as unverified. A failed release also records that the
group is gone, because torch clears its own record of the default group only *after* the shutdown that
failed — so `torch.distributed.is_initialized()` would otherwise keep answering `True` over a dead
communicator, and a second `train(multi_gpu=True)` in the same process would build on it.

**The collective timeout is set explicitly.** Torch's default for `nccl` is ten minutes, and at the end of
training the other processes wait while the main process computes the latent representation over the whole
dataset. On an atlas that single pass can take longer than ten minutes, and overrunning would abort the
waiting processes after training had already succeeded. `init_distributed` therefore asks for an hour, or for
whatever `NICHECOMPASS_COLLECTIVE_TIMEOUT_MINUTES` says.

**An empty data loader is rejected up front.** `drop_last` is set for distributed runs so that every process
performs the same number of iterations, which means a split shorter than one batch yields *nothing*. The node
loaders are consumed through a cycling generator that rebuilds its iterator when it runs out, so an empty one
is an infinite loop in pure Python — no collective is entered, no watchdog fires, and the run hangs with no
output and no error until its wall clock expires. Both the trainer and the generator now refuse it with a
message naming the fix.

**Two loss terms are not plain batch means, and they are handled explicitly.** The argument that N
processes reproduce one process rests on every loss term being a mean over the batch, so that a mean of
per-process means is the mean over the global batch. That holds only when every process averaged over the
*same number of items*, and two terms break it.

The **edge reconstruction loss** is the important one, because it carries `lambda_edge_recon` — the largest
weight in the model. When `cat_covariates_no_edges` drops the edges whose endpoints belong to different
categories of a covariate, the positives all survive (they are within-category) while the negatives are
sampled across the whole graph and mostly do not, so the number of surviving edges differs from process to
process. Both `pos_weight` and the mean's denominator then become per-process quantities. `compute_edge_recon_loss`
therefore sums the positive count, the negative count and the included-edge count across processes first,
computes the loss with `reduction="sum"`, and divides by the global count — with a factor of `world_size` to
undo the averaging the gradient reduction applies. The single-device path is untouched and still uses
`reduction="mean"`.

The **categorical covariates contrastive loss** cannot be repaired this way. It relabels the most and least
confident of the different-category edges and finds them with a `topk` over whatever edges the process was
given, so four processes each take the top fraction of their own quarter — and the union of four per-shard
quantiles is not the global quantile. A different set of edges is relabelled than one process would relabel;
the pseudo-labels themselves diverge, and no factor fixes that. `lambda_cat_covariates_contrastive > 0` with
`multi_gpu=True` therefore raises `NotImplementedError` rather than silently training a different model. It
defaults to `0.` in the reference pipeline, so this does not arise there.

**The wrapper is never stored on the model.** `self.model` stays the bare module and the
`DistributedDataParallel` wrapper lives only inside the trainer. This keeps `save`, `load` and every
`self.model.<attribute>` access working, and it means a checkpoint from a multi-GPU run has exactly the same
keys as one from a single-GPU run — the wrapper would otherwise prefix every key with `module.`.

## 6. Memory

Every process holds a full copy of the graph, the count matrix and `adata`. On one node with `N` GPUs the
host memory requirement is therefore roughly `N` times that of a single-device run, and host RAM rather
than GPU memory is usually what binds first on atlas-scale data. If a run fails with an out-of-memory error
that does not mention CUDA, this is the reason; use fewer processes.

Everything after training — the latent representation, the active gene program names, the optional
covariate embeddings and reconstructed edge probabilities — runs on the main process only, since each
process holds its own copy of `adata` and only the main one is kept.

## 7. Retrieve the prior gene program caches first

The gene program resources download and cache on first use. Under `torchrun` all processes reach that code
at once and would race to write the same cache files. Run the pipeline once on a single process to populate
the caches under `data/gene_programs/`, then launch the multi-GPU run.

## 8. What has been verified, and what has not

**Verified on four H100s.** A one-epoch run of the Xenium human breast cancer reference model
(`n_epochs 1`, 254,127 training nodes, 1,155,480 training edges, 313 genes, 131 prior + 100 add-on gene
programs) completed end to end on 4×H100 under LSF, `mode=exclusive_process`, launched with `mpirun`:
the `nccl` backend, per-process device binding by local rank, the gradient reduction, the all-reduced epoch
logs, the gathered validation metrics, the best-model reload, and the whole post-training path down to
`adata.write` and `model.save`. Reported `val AUROC 0.9543`, `AUPRC 0.9687`, target/source RNA MSE
1.2872 / 0.6459. Three consecutive runs gave AUROC 0.9552, 0.9552 and 0.9543 — the spread is expected, since
the processes draw different negative edges (see section 3).

**The tests** in `tests/test_distributed.py` run real multi-process training over the `gloo` backend on CPU
and check the gradient equivalence claim, the shard properties, the collective reductions, the rank helpers,
and every failure mode listed in section 5. They do not need a GPU, because the correctness of the split is
not a property of the device.

One known caveat that is not fixed in code: with `n_fc_layers_encoder=2` the encoder gains a `BatchNorm1d`,
whose running statistics are per process and are never synchronized (`broadcast_buffers=False`, and torch's
`broadcast_buffers=True` would broadcast rank 0's rather than average them). The checkpoint then holds one
process's statistics. This is stock behaviour for plain BatchNorm under data parallelism, whose documented
remedy is `nn.SyncBatchNorm.convert_sync_batchnorm`; the default `n_fc_layers_encoder=1` constructs no
BatchNorm at all, so it does not arise unless you ask for it.

**Measured: the per-epoch scaling, and it is modest under the global convention.** One epoch of the
reference configuration took 1 min 25 s on 4 GPUs and 1 min 39 s on 2. Both run the same 2,256 optimizer
steps, so the per-step cost is 37.7 ms at 128 edges per rank and 43.9 ms at 256. Solving `f + v·b` across
those two points gives a variable cost of 0.049 ms per edge per rank and a **fixed cost of 31.5 ms — 84% of
a 4-GPU step**.

That 84% is the loader sampling synchronously in the training process, the kernel launches, the gradient
all-reduce and the two per-iteration `.item()` synchronizations. Under `"global"` scaling it is paid 2,256
times per epoch whatever the device count, which is what caps the speedup: extrapolating to one device gives
roughly 127 s per epoch, so **four GPUs are only about 1.5× faster than one**.

The same fit predicts what `"per_process"` scaling would give, since it pays that fixed cost `world_size`
times less often: about 32 s per epoch on 4 GPUs, **2.7× better than the global convention on the same
hardware**, and about 16 s on 8. That is an extrapolation from two measurements rather than a measurement,
but it is the reason the per-process convention exists.

**Still not measured: whether a full-length run agrees with one GPU.** Results are deliberately not
bit-identical, and it is worth being precise about how far from identical, because the answer is "about as
far as two different seeds".

Four independent things differ between a run on one device and a run on several, and only the first is
usually mentioned:

1. **The sampled neighbourhoods.** `n_sampled_neighbors` is typically smaller than the degree of the
   spatial graph — the reference configuration samples 4 of 8 — so each batch sees a random half of every
   node's neighbourhood, redrawn every batch, in training and in validation. Each process draws its own.
   This is the largest of the four, and larger than the negative-edge resampling it is usually reduced to.
2. **The negative edges**, drawn per process at iteration time.
3. **The seed partition.** Seeds are split strided and disjoint, and each process shuffles only within its
   own shard, so "global step *k*" is not the set of edges a single-GPU run sees at step *k*, whatever the
   random state.
4. **`drop_last`.** It is on for distributed runs so that every process performs the same number of
   iterations, and off on one device. With 1,155,480 training edges and a global batch of 512 that is 2,257
   steps per epoch on one device against 2,256 on several — one extra optimizer step per epoch.

Re-seeding per rank does *not* put rank 0 back on the single-device stream either: by that point the
generator has been advanced by the train/validation split, so `seed + 0` resets it rather than continuing
it. Nothing sets `torch.use_deterministic_algorithms` or `cudnn.deterministic`, and the GATv2 encoder's
scatter aggregation on CUDA is atomics-ordered, so even identical inputs would not give identical outputs.

**What this means for comparing runs.** A single 1-GPU versus *N*-GPU comparison has no scale on its own.
Measure the noise floor first: two single-GPU runs differing only in `--seed`. If the multi-GPU gap sits
inside that spread, there is nothing device-count-specific to explain. Early stopping and
`reduce_lr_on_plateau` amplify any small trajectory difference — a 10-fold learning-rate cut firing at a
different epoch, then a different stopping epoch, then `reload_best_model` loading a different epoch's
weights — so a controlled comparison should disable both (`use_early_stopping=False`,
`reload_best_model=False`) and compare at a fixed epoch.

For reproducing published results, train on one GPU.
