"""
optimization.py

General utilities for optimization, e.g., schedulers such as Linear Warmup w/ Cosine Decay for Transformer training.
Notably *does not* use the base PyTorch LR Scheduler, since we call it continuously, across epochs, across steps;
PyTorch has no built-in way of separating the two without coupling to the DataLoader, so may as well make this explicit
in the parent loop.

References
    - MAE: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_sched.py
    - ⚡️-Bolts: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/optimizers/lr_scheduler.py
"""
import math
from typing import Callable

from torch.optim.optimizer import Optimizer


def get_lr_update(
    opt: Optimizer, schedule: str, lr: float, min_lr: float, warmup_epochs: int, max_epochs: int
) -> Callable[[int, float], float]:
    if schedule == "linear-warmup+cosine-decay":

        def lr_update(epoch: int, fractional_progress: float) -> float:
            """Run the warmup check for linear increase, else cosine decay."""
            if (epoch + fractional_progress) < warmup_epochs:
                new_lr = lr * (epoch + fractional_progress) / max(1.0, warmup_epochs)
            else:
                # Cosine Decay --> as defined in the SGDR Paper...
                progress = ((epoch + fractional_progress) - warmup_epochs) / max(1.0, max_epochs - warmup_epochs)
                new_lr = min_lr + (lr - min_lr) * (0.5 * (1 + math.cos(math.pi * progress)))

            # Apply...
            for group in opt.param_groups:
                if "lr_scale" in group:
                    group["lr"] = new_lr * group["lr_scale"]
                else:
                    group["lr"] = new_lr

            return new_lr

    elif schedule == "none":

        def lr_update(_: int, __: float) -> float:
            return lr

    else:
        raise NotImplementedError(f"Schedule `{schedule}` not implemented!")

    return lr_update
