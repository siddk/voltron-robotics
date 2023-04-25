"""
distributed.py

Key distributed utilities; notably provides a standard API for getting relevant data from either CPU/GPU or XLA (TPU)
devices, since the underlying implementation does differ substantially.

Assumes that code is running with one of the following strategies:
    - Single Process (on CPU, GPU)
    - DDP (CPU, GPU)... uses the torch.distributed.launch API & semantics
    - XMP Spawn (TPU)... TPU based XLA + Multiprocessing Spawn semantics

Key Terminology
    -> World Size :: Total number of processes distributed over (# nodes x # devices) -- assumed homogenous!
    -> Rank :: Integer index of current process in the total world size
    -> Local Rank :: Local index on given node in [0, Devices per Node]
"""
from importlib.util import find_spec
from typing import Iterator, TypeVar

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

T_co = TypeVar("T_co", covariant=True)


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


def xla_available() -> bool:
    try:
        return find_spec("torch_xla") is not None
    except ModuleNotFoundError:
        return False


def get_rank() -> int:
    """Returns the global rank [0, World Size) of the current process."""
    if xla_available():
        import torch_xla.core.xla_model as xm

        # By default, if XLA is available, assume we're running under XMP Spawn
        return xm.get_ordinal()

    # Try to get rank via torch.distributed, but catch error if only single process
    try:
        return torch.distributed.get_rank()

    # RuntimeError => not running distributed (single process)
    except RuntimeError:
        return 0
