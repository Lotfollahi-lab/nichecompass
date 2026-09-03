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
        # Dispersion parameters, standing in for target_rna_theta and
        # source_rna_theta. NicheCompass uses these in the LOSS rather than in
        # the forward pass, which is what makes where the loss is computed
        # matter to DistributedDataParallel.
        self.target_theta = nn.Parameter(torch.randn(n_features))
        self.source_theta = nn.Parameter(torch.randn(n_features))

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
    def __init__(self, model: nn.Module, edge_recon_active: bool=True):
        super().__init__()
        self.model = model
        # Off for the first epochs of a real run, which is what makes
        # global_loss and optim_loss cover different parameters
        self.edge_recon_active = edge_recon_active

    def forward(self, node_x, edge_x, node_y, edge_y):
        """
        Both passes AND the loss, so that every use of a parameter falls inside
        the one forward that ´DistributedDataParallel´ sees. The loss uses the
        dispersion parameters and the decoder weights, so computing it outside
        makes the reducer mark those parameters ready twice.

        Only ´optim_loss´ leaves this forward with its graph. The other entries
        are detached, because ´find_unused_parameters´ decides which parameters
        the reducer waits for by walking the graphs of everything the forward
        returns, and ´global_loss´ holds a term that ´optim_loss´ omits while
        it is warming up.
        """
        node_out = self.model(node_x, "omics")
        edge_out = self.model(edge_x, "graph")
        loss_dict = loss_fn(node_out, edge_out, node_y, edge_y, self.model,
                            edge_recon_active=self.edge_recon_active)
        return {key: (value if key == "optim_loss" else value.detach())
                for key, value in loss_dict.items()}


def loss_fn(node_out, edge_out, node_y, edge_y, model,
            edge_recon_active: bool=True) -> dict:
    """
    A dict of loss terms, as the NicheCompass loss returns. Means over the
    batch, as every NicheCompass loss term is, plus terms that use parameters
    directly: the dispersion parameters, standing in for ´theta´, and an L1
    penalty on the decoder weights, standing in for the gene program
    regularizers.

    ´global_loss´ and ´optim_loss´ differ exactly as they do in NicheCompass:
    the edge reconstruction term is reported in the former from the first
    epoch, and enters the latter only once it is switched on.
    """
    node_recon = -(node_y * torch.exp(model.target_theta)
                   * torch.log(node_out + 1e-8)).sum(-1).mean()
    edge_recon = -(edge_y * torch.exp(model.source_theta)
                   * torch.log(edge_out + 1e-8)).sum(-1).mean()
    regularization = 1e-3 * (model.target_decoder.weight.abs().sum()
                             + model.source_decoder.weight.abs().sum())

    loss_dict = {"node_recon_loss": node_recon,
                 "edge_recon_loss": edge_recon,
                 "reg_loss": regularization}
    loss_dict["global_loss"] = node_recon + edge_recon + regularization
    loss_dict["optim_loss"] = node_recon + regularization
    if edge_recon_active:
        loss_dict["optim_loss"] = loss_dict["optim_loss"] + edge_recon
    return loss_dict


def gradients_of(model: nn.Module) -> dict:
    """
    A gradient per parameter, with an absent gradient reported as zeros. A
    parameter that a warming-up loss term does not reach has no ´.grad´ at all,
    on one process and on several alike, and that is precisely the parameter a
    reducer would be left waiting for.
    """
    return {name: (parameter.grad.clone() if parameter.grad is not None
                   else torch.zeros_like(parameter))
            for name, parameter in model.named_parameters()}


def make_batch(seed: int, n_obs: int, n_features: int=6):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n_obs, n_features, generator=generator)
    y = torch.rand(n_obs, n_features, generator=generator)
    return x, y / y.sum(-1, keepdim=True)


def single_process_gradients(n_obs: int,
                             edge_recon_active: bool=True) -> dict:
    """Gradients of one process over the whole global batch."""
    torch.manual_seed(0)
    model = TwoDecoderModel()
    node_x, node_y = make_batch(1, n_obs)
    edge_x, edge_y = make_batch(2, n_obs)
    JointForward(model, edge_recon_active=edge_recon_active)(
        node_x, edge_x, node_y, edge_y)["optim_loss"].backward()
    return gradients_of(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_obs", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    # The warm-up state of a real run: reported in global_loss, not yet
    # optimized. A run in this state is the one that exposes a reducer waiting
    # for gradients the backward pass never produces.
    parser.add_argument("--no_edge_recon", action="store_true")
    args = parser.parse_args()
    args.edge_recon_active = not args.no_edge_recon

    if not is_distributed_launch():
        sys.exit("the worker was launched without the distributed environment")
    init_distributed(backend="gloo")
    rank, world_size = get_rank(), get_world_size()

    torch.manual_seed(0)
    model = TwoDecoderModel()
    ddp_model = nn.parallel.DistributedDataParallel(
        JointForward(model, edge_recon_active=args.edge_recon_active),
        find_unused_parameters=True)

    # The same global batch as a single process run, split across processes
    node_x, node_y = make_batch(1, args.n_obs)
    edge_x, edge_y = make_batch(2, args.n_obs)
    shard = shard_indices(torch.arange(args.n_obs))
    loss_dict = ddp_model(node_x[shard], edge_x[shard],
                          node_y[shard], edge_y[shard])
    loss_dict["optim_loss"].backward()
    # Snapshot before the second iteration, which accumulates into .grad, so
    # that the comparison is against a single backward pass
    gradients = gradients_of(model)

    # The quantity that drives gene program pruning: every process has to end
    # up with the same value, or the processes prune different gene programs
    # and silently train different models
    reduced = torch.full((3,), float(rank + 1))
    all_reduce_mean(reduced)

    # The validation predictions each process accumulates for its own shard,
    # deliberately of different lengths per process
    gathered = all_gather_numpy(np.array([float(rank)] * (rank + 2)),
                                torch.device("cpu"))

    # A SECOND iteration, because a reducer that is still waiting for gradients
    # from the first one only raises when the next forward starts
    model.zero_grad(set_to_none=True)
    second = ddp_model(node_x[shard], edge_x[shard],
                       node_y[shard], edge_y[shard])
    second["optim_loss"].backward()

    torch.save({"rank": rank,
                "world_size": world_size,
                "is_main": is_main_process(),
                "unwraps": unwrap_model(ddp_model) is not ddp_model,
                "shard": shard,
                "reduced": reduced,
                "gathered": torch.as_tensor(gathered),
                "gradients": gradients},
               os.path.join(args.out_dir, f"rank_{rank}.pt"))
    cleanup_distributed()


if __name__ == "__main__":
    main()
