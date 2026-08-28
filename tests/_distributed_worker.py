"""
One process of a distributed test run.

Kept in its own file, and launched as a subprocess with the environment
variables that ´torchrun´ sets, so that the tests exercise the real
´init_distributed´ code path rather than a hand rolled process group. Launching
subprocesses also avoids spawning from inside pytest, which deadlocks on
platforms whose default start method re-imports the test module.

Not a test module itself: the leading underscore keeps pytest from collecting
it.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from nichecompass.train.distributed import (all_gather_numpy,
                                            all_reduce_mean,
                                            cleanup_distributed,
                                            get_rank,
                                            get_world_size,
                                            init_distributed,
                                            is_distributed_launch,
                                            is_main_process,
                                            shard_indices,
                                            unwrap_model)


class TwoDecoderModel(nn.Module):
    """
    A small stand-in for the NicheCompass module that keeps the three
    structural properties the distributed path depends on: one latent shared by
    two decoders, a softmax over all features in each decoder, and losses that
    are means over the batch.
    """
    def __init__(self, n_features: int=6, n_gps: int=4):
        super().__init__()
        self.encoder = nn.Linear(n_features, n_gps, bias=False)
        self.target_decoder = nn.Linear(n_gps, n_features, bias=False)
        self.source_decoder = nn.Linear(n_gps, n_features, bias=False)

    def forward(self, x, decoder):
        z = self.encoder(x)
        layer = (self.target_decoder if decoder == "omics"
                 else self.source_decoder)
        return torch.softmax(layer(z), dim=-1)


class JointForward(nn.Module):
    """
    The joining of the node-level and the edge-level pass that the trainer
    performs, so that ´DistributedDataParallel´ sees one forward per backward.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, node_x, edge_x):
        return (self.model(node_x, "omics"), self.model(edge_x, "graph"))


def loss_fn(node_out, edge_out, node_y, edge_y):
    """Means over the batch, as every NicheCompass loss term is."""
    return (-(node_y * torch.log(node_out + 1e-8)).sum(-1).mean()
            + -(edge_y * torch.log(edge_out + 1e-8)).sum(-1).mean())


def make_batch(seed: int, n_obs: int, n_features: int=6):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n_obs, n_features, generator=generator)
    y = torch.rand(n_obs, n_features, generator=generator)
    return x, y / y.sum(-1, keepdim=True)


def single_process_gradients(n_obs: int) -> dict:
    """Gradients of one process over the whole global batch."""
    torch.manual_seed(0)
    model = TwoDecoderModel()
    node_x, node_y = make_batch(1, n_obs)
    edge_x, edge_y = make_batch(2, n_obs)
    node_out, edge_out = JointForward(model)(node_x, edge_x)
    loss_fn(node_out, edge_out, node_y, edge_y).backward()
    return {name: parameter.grad.clone()
            for name, parameter in model.named_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_obs", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    if not is_distributed_launch():
        sys.exit("the worker was launched without the distributed environment")
    init_distributed(backend="gloo")
    rank, world_size = get_rank(), get_world_size()

    torch.manual_seed(0)
    model = TwoDecoderModel()
    ddp_model = nn.parallel.DistributedDataParallel(
        JointForward(model), find_unused_parameters=True)

    # The same global batch as a single process run, split across processes
    node_x, node_y = make_batch(1, args.n_obs)
    edge_x, edge_y = make_batch(2, args.n_obs)
    shard = shard_indices(torch.arange(args.n_obs))
    node_out, edge_out = ddp_model(node_x[shard], edge_x[shard])
    loss_fn(node_out, edge_out, node_y[shard], edge_y[shard]).backward()

    # The quantity that drives gene program pruning: every process has to end
    # up with the same value, or the processes prune different gene programs
    # and silently train different models
    reduced = torch.full((3,), float(rank + 1))
    all_reduce_mean(reduced)

    # The validation predictions each process accumulates for its own shard,
    # deliberately of different lengths per process
    gathered = all_gather_numpy(np.array([float(rank)] * (rank + 2)),
                                torch.device("cpu"))

    torch.save({"rank": rank,
                "world_size": world_size,
                "is_main": is_main_process(),
                "unwraps": unwrap_model(ddp_model) is not ddp_model,
                "shard": shard,
                "reduced": reduced,
                "gathered": torch.as_tensor(gathered),
                "gradients": {name: parameter.grad.clone() for name, parameter
                              in model.named_parameters()}},
               os.path.join(args.out_dir, f"rank_{rank}.pt"))
    cleanup_distributed()


if __name__ == "__main__":
    main()
