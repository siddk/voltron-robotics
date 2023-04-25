"""
checkpointing.py

Core utility class for handling model/optimizer serialization & checkpointing -- including resume from checkpoint logic.

Support the following strategies:
    - (k, -1, -1) --> Keep only the most recent "k" epoch checkpoints
    - (k, m, -1) --> Keep the most recent "k" epoch checkpoints and *every* m epoch checkpoint
    - (k, m, s = 2500) --> Keep "k" and "m" subject to above, but also keep *s* step checkpoints for current epoch
"""
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

# Grab Logger
overwatch = logging.getLogger(__file__)


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


class CheckpointSaver:
    def __init__(self, strategy: Tuple[int, int, int], run_dir: str, is_rank_zero: bool = False) -> None:
        """
        Create a checkpoint saver with the provided strategy that saves to the given path.

        :param strategy: Strategy, following the (k, -1, -1) -- (k, m, -1) -- (k, m, s) description above.
        :param run_dir: Path to root of `run_dir`.
        :param is_rank_zero: Boolean whether this process is global zero (no-op if not)!
        """
        (self.k, self.m, self.s), self.run_dir, self.is_rank_zero = strategy, run_dir, is_rank_zero
        self.recents, self.intervals, self.step_checkpoints = FixedDeck(maxlen=self.k), set(), set()

        # If `self.s == -1` --> *Disable* step checkpoints (only at save end of epoch!)
        self.enable_step = self.s != -1

        # Create "checkpoints" subdirectory
        self.path = Path(run_dir) / "checkpoints"
        if self.is_rank_zero:
            os.makedirs(self.path, exist_ok=True)

            # Populate `step_checkpoints` on __init__ (if resuming *within* an epoch!)
            self.step_checkpoints.update([c for c in self.path.iterdir() if "local-epoch=" in str(c)])

        # Created Saver...
        overwatch.info(f"Created CheckpointSaver with `k = {self.k}` -- `m = {self.m}` -- s = {self.s}!")

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
        """Performs a global zero save operation, unlinking stale checkpoints if necessary."""
        if not self.is_rank_zero:
            return

        # Check if saving a `local_step` (within an epoch) or if end of epoch...
        if self.enable_step and is_local_step and (local_step % self.s) == 0:
            step_checkpoint = self.path / f"local-epoch={epoch}-step={local_step}-t={duration}.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, step_checkpoint
            )

            # Update Relevant Trackers...
            self.step_checkpoints.add(step_checkpoint)

        elif not is_local_step:
            if train_loss is None and val_loss is None:
                checkpoint = self.path / f"epoch={epoch}-train=inf-val=inf-t={duration}.pt"
            else:
                checkpoint = self.path / f"epoch={epoch}-train={train_loss:.4f}-val={val_loss:.4f}-t={duration}.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoint
            )

            # Update Relevant Trackers
            if epoch % self.m == 0:
                self.intervals.add(checkpoint)

            # Remove all "step_checkpoints" now that we've made it to the end of an epoch!
            while len(self.step_checkpoints) > 0:
                os.remove(self.step_checkpoints.pop())

            # Add to recents & flush stale checkpoints...
            to_remove = self.recents.append(checkpoint)
            if to_remove is not None and to_remove not in self.intervals:
                os.remove(to_remove)


def do_resume(resume: bool, run_dir: str) -> Tuple[Optional[Path], int, int]:
    """Handle `resume` logic --> consists of retrieving checkpoint_path and epoch/step computation (if resuming)."""
    if not resume:
        # We're starting a fresh run --> return None for checkpoint_path, resume_epoch = 0, resume_step = 0
        return None, 0, 0

    # === Auto-Resume Logic ===
    # **IMPORTANT**: We're making a few assumptions on resuming that should eventually become explicit checks:
    #   - `accumulate_grad_batches` is exactly the same when resuming; this means:
    #       + `model_cfg.effective_bsz`, `model_cfg.fabric_bsz`, & `accelerator_cfg.num_accelerators` are the same!
    #   - The Weights & Biases directory `run_dir/wandb` only contains a *single run*
    #   - The `param_groups` in `optimizer.state_dict()` are exactly the same across resumes!
    #       + This means that (and generally should be true for resuming altogether) the architecture is the same!
    #   - The `cfg.seed` should be the same (again, should generally be true...)
    all_checkpoints_path, resume_checkpoint, resume_epoch, resume_step = Path(run_dir) / "checkpoints", None, 0, 0
    if all_checkpoints_path.exists() and any(all_checkpoints_path.iterdir()):
        # Parse out the latest "complete" epoch checkpoint, as well as any "local step" checkpoints...
        checkpoints = list(all_checkpoints_path.iterdir())
        complete_checkpoint, complete_epoch = max(
            [
                (c, int(re.search("epoch=(.+?)-train", c.name).group(1)))
                for c in checkpoints
                if "local-epoch=" not in str(c)
            ],
            key=lambda x: x[1],
        )

        # Case 1 :: We have "local step" checkpoints --> will always override any "full epoch" checkpoints...
        local = [
            (
                c,
                int(re.search("local-epoch=(.+?)-step", c.name).group(1)),
                int(re.search("step=(.+?)[.-]", c.name).group(1)),
            )
            for c in checkpoints
            if "local-epoch=" in str(c)
        ]
        if len(local) > 0:
            # Parse out (epoch, "highest" step) + assert no great "full epoch" checkpoint exists!
            resume_checkpoint, resume_epoch, resume_step = max(local, key=lambda x: x[1:])
            assert resume_epoch == complete_epoch, "Epoch mismatch in `resume` from local_step!"

        # Case 2 :: Otherwise, we're just going to start with the last "complete" epoch
        else:
            resume_checkpoint, resume_epoch = complete_checkpoint, complete_epoch

    return resume_checkpoint, resume_epoch, resume_step
