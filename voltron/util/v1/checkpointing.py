"""
checkpointing.py

XLA-specific utility class for handling model/optimizer serialization & checkpointing.

Support the following strategies:
    - (k, -1, -1) --> Keep only the most recent "k" epoch checkpoints
    - (k, m, -1) --> Keep the most recent "k" epoch checkpoints and *every* m epoch checkpoint
    - (k, m, s = 2500) --> Keep "k" and "m" subject to above, but also keep *s* step checkpoints for current epoch
"""
import os
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple

import torch.nn as nn
from torch.optim.optimizer import Optimizer


class FixedDeck(deque):
    def __init__(self, maxlen: int) -> None:
        super().__init__(maxlen=maxlen)

    def append(self, x: Any) -> Any:
        pop_value = None
        if self.__len__() == self.maxlen:
            pop_value = self.__getitem__(0)

        # Perform parent append and return popped value, if any!
        super().append(x)
        return pop_value


class XLACheckpointSaver:
    def __init__(self, strategy: Tuple[int, int, int], run_dir: str) -> None:
        """
        Create a checkpoint saver with the provided strategy that saves to the given path, with XLA-specific handling.

        :param strategy: Strategy, following the (k, -1, -1) -- (k, m, -1) -- (k, m, s) description above.
        :param run_dir: Path to root of `run_dir`
        """
        import torch_xla.core.xla_model as xm

        (self.k, self.m, self.s), self.run_dir = strategy, run_dir
        self.recents, self.intervals, self.step_checkpoints = FixedDeck(maxlen=self.k), set(), set()

        # If `self.s` is -1 --> disable step_checkpoints
        self.enable_step = self.s != -1

        # Create "checkpoints" subdirectory
        self.path = Path(run_dir) / "checkpoints"
        if xm.is_master_ordinal(local=False):
            os.makedirs(self.path, exist_ok=True)

        # Populate `step_checkpoints` on __init__ (if resuming *within* an epoch...)
        self.step_checkpoints.update([c for c in self.path.iterdir() if "local-epoch=" in str(c)])

        # Create Saver
        xm.master_print(f"Created Saver w/ `k` = {self.k}, `m` = {self.m}`, `s` = {self.s}!")

    def save(
        self,
        epoch: int,
        is_local_step: bool,
        model: nn.Module,
        optimizer: Optimizer,
        duration: int,
        local_step: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> None:
        """Performs the save operation, unlinking existing stale checkpoints, if necessary."""
        import torch_xla.core.xla_model as xm

        # Check if saving a `local_step` (within an epoch) or if saving an `epoch`
        if self.enable_step and is_local_step and (local_step % self.s) == 0:
            # Create filename
            step_checkpoint = self.path / f"local-epoch={epoch}-step={local_step}-t={duration}.pt"

            # Perform actual save action...
            #   > IMPORTANT --> XLA/XM will throw an error if optimizer has "param_groups" so only save "state"...
            xm.save([model.state_dict(), optimizer.state_dict()["state"]], step_checkpoint)
            if xm.is_master_ordinal(local=False):
                self.step_checkpoints.add(step_checkpoint)

        elif not is_local_step:
            # Create filename
            if train_loss is None and val_loss is None:
                checkpoint = self.path / f"epoch={epoch}-train=inf-val=inf-t={duration}.pt"
            else:
                checkpoint = self.path / f"epoch={epoch}-train={train_loss:.4f}-val={val_loss:.4f}-t={duration}.pt"

            # Perform actual save action...
            #   > IMPORTANT --> XLA/XM will throw an error if optimizer has "param_groups" so only save "state"...
            xm.save([model.state_dict(), optimizer.state_dict()["state"]], checkpoint)

            if xm.is_master_ordinal(local=False):
                # Conditional Check for M -- Keep if modulated by interval
                if epoch % self.m == 0:
                    self.intervals.add(checkpoint)

                # Remove all "step_checkpoints" now that we successfully made it to the end of the epoch!
                while len(self.step_checkpoints) > 0:
                    os.remove(self.step_checkpoints.pop())

                # Finally, recency add & unlink/delete if necessary
                to_remove = self.recents.append(checkpoint)
                if to_remove is not None and to_remove not in self.intervals:
                    os.remove(to_remove)
