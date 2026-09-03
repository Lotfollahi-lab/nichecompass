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

**Slurm, one node** (`sbatch submit_slurm.sh`):

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train_nichecompass_reference_model.py --multi_gpu <other arguments>
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

**The batch sizes stay global.** `edge_batch_size` and `node_batch_size` mean the same thing on one device
and on eight. With `world_size` processes each one takes a `world_size`-th of every batch, so the number of
optimizer steps per epoch is unchanged and the speedup comes from dividing the work of each step rather
than from making the step bigger.

This matters for more than convenience. Gene program pruning starts at epoch `n_epochs_all_gps` and is
driven by an exponential moving average updated once per optimizer step. Had the per-process batch been
kept constant instead, an epoch would have contained `world_size` times fewer steps, the moving average
would have seen `world_size` times fewer updates by the time pruning starts, and a multi-GPU run would have
pruned different gene programs than a single-GPU run of the same configuration.

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

## 5. Five structural details

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

## 8. What has not been verified

The tests in `tests/test_distributed.py` run real multi-process training over the `gloo` backend on CPU and
check the gradient equivalence claim, the shard properties, the collective reductions and the rank helpers.
They do not need a GPU, because the correctness of the split is not a property of the device.

Not covered, because they need a machine with several GPUs: the `nccl` backend, binding each process to its
device by local rank, and the actual speedup. These follow standard PyTorch practice, but they are
untested here and should be confirmed on the target machine before a long run.
