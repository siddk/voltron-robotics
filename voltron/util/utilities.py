"""
utilities.py

General utilities for randomness, distributed training, and miscellaneous checks in PyTorch.

=== Randomness ===

Random `seed_everything` functionality is taken directly from PyTorch-Lighting:
    > Ref: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/seed.py

This is pretty important to get right if we're every randomly generating our masks (or prefix dropout) inside our
Dataset __getitem__() with multiple workers... if not handled properly, we will get repeated augmentations anytime
we inject randomness from non-PyTorch sources (e.g., numpy, random)!
    > Ref: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

=== Distributed / DDP Training ====

Utilities provide a standard API across single-GPU/multi-GPU/multi-node training. Assumes that code is running with
one of the following strategies:
    - Single Process (on CPU?, GPU)
    - DDP (GPU, Multi-Node GPU) --> uses the `torchrun`/`torch.distributed` API & semantics

Key Terminology
    -> World Size :: Total number of processes distributed over (# nodes x # devices) -- assumed homogenous!
    -> Rank :: Integer index of current process in the total world size
    -> Local Rank :: Local index on given node in [0, Devices per Node]
"""
import os
import random
from typing import Callable, Iterator, Optional, TypeVar

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

T_co = TypeVar("T_co", covariant=True)


# === Randomness ===


def worker_init_function(worker_id: int) -> None:
    """
    Borrowed directly from PyTorch-Lightning; inspired by this issue comment in the PyTorch repo:
        > Ref: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Intuition: You can think of the seed sequence spawn function as a "janky" torch.Generator() or jax.PRNGKey that
    you can run iterative splitting on to get new (predictable) randomness.

    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    # Get current `rank` (if running distributed) and `process_seed`
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None


# === Distributed Training ===


class ResumeableDistributedSampler(DistributedSampler):
    def __init__(
        self,
        seen_examples: int,
        resume_epoch: int,
        dataset: Dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.seen_examples, self.resume_epoch, self.do_resume = seen_examples, resume_epoch, True

        # Set `seen_examples_per_replica` --> this is necessary for when we re-wrap the iterator in self.__iter__()
        #   > Note: `seen_examples` is across _all_ replicas --> so divide!
        self.seen_examples_per_replica = self.seen_examples // self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        epoch_iterator = super().__iter__()
        if self.do_resume:
            # Unpack iterator --> list, slice off the first `seen_examples_per_replica` examples, and re-wrap!
            leftover_idxs = list(epoch_iterator)[self.seen_examples_per_replica :]
            return iter(leftover_idxs)
        else:
            return epoch_iterator

    def __len__(self) -> int:
        if self.do_resume:
            # Remove the "seen" sample from self.num_samples; num_samples is *per replica*!
            return self.num_samples - self.seen_examples_per_replica
        else:
            return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        # If epoch != self.resume_epoch --> we're in "regular DistributedSampler" mode (just a wrapper class)
        #   > Intuition: We should *only* truncate examples on the first epoch upon resuming!
        self.epoch = epoch
        if self.epoch != self.resume_epoch:
            self.do_resume = False
