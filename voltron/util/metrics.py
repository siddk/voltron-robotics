"""
metrics.py

Utility classes defining Metrics containers with model-specific logging to various endpoints (JSONL local logs, W&B).
"""
import os
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jsonlines
import numpy as np
import torch
import wandb

from voltron.conf import TrackingConfig

# === Define Loggers (`Logger` is an abstract base class) ===


class Logger(ABC):
    def __init__(self, run_id: str, hparams: Dict[str, Any], is_rank_zero: bool = False) -> None:
        self.run_id, self.hparams, self.is_rank_zero = run_id, hparams, is_rank_zero

    @abstractmethod
    def write_hyperparameters(self) -> None:
        raise NotImplementedError("Logger is an abstract class!")

    @abstractmethod
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        raise NotImplementedError("Logger is an abstract class!")

    def finalize(self) -> None:
        time.sleep(1)


class JSONLinesLogger(Logger):
    def write_hyperparameters(self) -> None:
        if not self.is_rank_zero:
            return

        # Only log if `is_rank_zero`
        with jsonlines.open(f"{self.run_id}.jsonl", mode="w", sort_keys=True) as js_logger:
            js_logger.write(
                {
                    "run_id": self.run_id,
                    "start_time": datetime.now().strftime("%m-%d-%H:%M"),
                    "hparams": self.hparams,
                }
            )

    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        if not self.is_rank_zero:
            return

        # Only log if `is_rank_zero`
        with jsonlines.open(f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
            js_logger.write(metrics)


class WeightsBiasesLogger(Logger):
    def __init__(
        self,
        run_id: str,
        hparams: Dict[str, Any],
        tracking_cfg: TrackingConfig,
        tags: List[str],
        resume: bool = False,
        resume_id: Optional[str] = None,
        is_rank_zero: bool = False,
    ) -> None:
        super().__init__(run_id, hparams, is_rank_zero)
        self.tracking_cfg, self.tags, self.resume, self.resume_id = tracking_cfg, tags, resume, resume_id
        self.path = Path(os.getcwd() if self.tracking_cfg.directory is None else self.tracking_cfg.directory)

        # Handle (Automatic) Resume if `resume = True`
        if self.resume and self.resume_id is None:
            wandb_path = self.path / "wandb"
            if wandb_path.exists() and any((wandb_path / "latest-run").iterdir()):
                # Parse unique `run_id` from the `.wandb.` file...
                wandb_fns = [f.name for f in (wandb_path / "latest-run").iterdir() if f.name.endswith(".wandb")]
                assert len(wandb_fns) == 1, f"There should only be 1 `.wandb.` file... found {len(wandb_fns)}!"

                # Regex Match on `run-{id}.wandb`
                self.resume_id = re.search("run-(.+?).wandb", wandb_fns[0]).group(1)

            elif wandb_path.exists():
                raise ValueError("Starting Training from Scratch with Preexisting W&B Directory; Remove to Continue!")

        # Call W&B.init()
        self.initialize()

    def initialize(self) -> None:
        """Run W&B.init on the guarded / rank-zero process."""
        if not self.is_rank_zero:
            return

        # Only initialize / log if `is_rank_zero`
        wandb.init(
            project=self.tracking_cfg.project,
            entity=self.tracking_cfg.entity,
            config=self.hparams,
            name=self.run_id,
            dir=self.path,
            tags=self.tags,
            notes=self.tracking_cfg.notes,
            resume="allow" if self.resume else False,
            id=self.resume_id,
        )

    def write_hyperparameters(self) -> None:
        if not self.is_rank_zero:
            return

        # Only log if `is_rank_zero`
        wandb.config = self.hparams

    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        if not self.is_rank_zero:
            return

        # Only log if `is_rank_zero`
        wandb.log(metrics, step=global_step)

    def finalize(self) -> None:
        wandb.finish()
        time.sleep(150)


# === Core Metrics Container :: Responsible for Initializing Loggers and Compiling/Pushing Metrics ===


class Metrics:
    def __init__(
        self,
        active_loggers: List[str],
        run_id: str,
        hparams: Dict[str, Any],
        model_arch: str,
        is_rank_zero: bool,
        tracking_cfg: Optional[TrackingConfig] = None,
        tags: Optional[List[str]] = None,
        resume: bool = False,
        resume_id: Optional[str] = None,
        window: int = 128,
    ) -> None:
        """High-Level Container Logic for Metrics Logging; logic defined for each model architecture!"""
        self.model_arch, self.is_rank_zero, self.window = model_arch, is_rank_zero, window

        # Initialize Loggers
        self.loggers = []
        for log_type in active_loggers:
            if log_type == "jsonl":
                logger = JSONLinesLogger(run_id, hparams, is_rank_zero=is_rank_zero)
            elif log_type == "wandb":
                logger = WeightsBiasesLogger(
                    run_id, hparams, tracking_cfg, tags, resume, resume_id, is_rank_zero=is_rank_zero
                )
            else:
                raise ValueError(f"Logger `{log_type}` is not defined!")

            # Add Hyperparameters --> Add to `self.loggers`
            logger.write_hyperparameters()
            self.loggers.append(logger)

        # Create Universal Trackers
        self.global_step, self.start_time, self.resume_time, self.step_start_time = 0, time.time(), 0, time.time()
        self.tracker = {
            "loss": deque(maxlen=self.window),
            "lr": [],
            "step_time": deque(maxlen=self.window),
        }

        # Create Model-Specific Trackers
        if self.model_arch == "v-mvp":
            self.tracker.update({"reconstruction_loss": deque(maxlen=self.window)})

        elif self.model_arch in {"v-r3m", "v-rn3m"}:
            self.tracker.update(
                {
                    "tcn_loss": deque(maxlen=self.window),
                    "reward_loss": deque(maxlen=self.window),
                    "l1_loss": deque(maxlen=self.window),
                    "l2_loss": deque(maxlen=self.window),
                    "tcn_accuracy": deque(maxlen=self.window),
                    "reward_accuracy": deque(maxlen=self.window),
                }
            )

        elif self.model_arch == "v-cond":
            self.tracker.update({"reconstruction_loss": deque(maxlen=self.window)})

        elif self.model_arch == "v-dual":
            self.tracker.update(
                {
                    "reconstruction_loss": deque(maxlen=self.window),
                    "zero_reconstruction_loss": deque(maxlen=self.window),
                    "k_reconstruction_loss": deque(maxlen=self.window),
                }
            )

        elif self.model_arch == "v-gen":
            self.tracker.update(
                {
                    "reconstruction_loss": deque(maxlen=self.window),
                    "zero_reconstruction_loss": deque(maxlen=self.window),
                    "k_reconstruction_loss": deque(maxlen=self.window),
                    "lm_loss": deque(maxlen=self.window),
                    "lm_ppl": deque(maxlen=self.window),
                }
            )

        else:
            raise ValueError(f"Metrics for Model `{self.model_arch}` are not implemented!")

    def itemize(self) -> Dict[str, torch.Tensor]:
        """Utility method for converting `deque[torch.Tensor] --> mean over Tensors."""
        return {
            k: torch.stack(list(v)).mean().item()
            for k, v in self.tracker.items()
            if k not in {"loss", "lr", "step_time"}
        }

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        for logger in self.loggers:
            logger.write(global_step, metrics)

    def finalize(self) -> None:
        for logger in self.loggers:
            logger.finalize()

    def get_status(self, epoch: int, loss: Optional[torch.Tensor] = None) -> str:
        lr = self.tracker["lr"][-1] if len(self.tracker["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Epoch {epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status!
        return f"=>> [Epoch {epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f} -- Loss :: {loss:.4f}"

    def commit(
        self,
        *,
        global_step: Optional[int] = None,
        resume_time: Optional[int] = None,
        lr: Optional[float] = None,
        update_step_time: bool = False,
        **kwargs,
    ) -> None:
        """Update all metrics in `self.tracker` by iterating through special positional arguments & kwargs."""
        if not self.is_rank_zero:
            return

        # Special Positional Arguments
        if global_step is not None:
            self.global_step = global_step

        if resume_time is not None:
            self.resume_time = resume_time

        if lr is not None:
            self.tracker["lr"].append(lr)

        if update_step_time:
            self.tracker["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            self.tracker[key].append(value.detach())

    def push(self, epoch: int) -> str:
        """Push current metrics to loggers with model-specific handling."""
        if not self.is_rank_zero:
            return

        loss = torch.stack(list(self.tracker["loss"])).mean().item()
        step_time, lr = np.mean(list(self.tracker["step_time"])), self.tracker["lr"][-1]
        status = self.get_status(epoch, loss)

        # Model-Specific Handling
        itemized = self.itemize()
        if self.model_arch == "v-mvp":
            self.log(
                self.global_step,
                metrics={
                    "Pretrain/Step": self.global_step,
                    "Pretrain/Epoch": epoch,
                    "Pretrain/V-MVP Train Loss": loss,
                    "Pretrain/Reconstruction Loss": itemized["reconstruction_loss"],
                    "Pretrain/Learning Rate": lr,
                    "Pretrain/Step Time": step_time,
                },
            )

        elif self.model_arch in {"v-r3m", "v-rn3m"}:
            self.log(
                self.global_step,
                metrics={
                    "Pretrain/Step": self.global_step,
                    "Pretrain/Epoch": epoch,
                    f"Pretrain/V-{'R3M' if self.model_arch == 'v-r3m' else 'RN3M'} Train Loss": loss,
                    "Pretrain/TCN Loss": itemized["tcn_loss"],
                    "Pretrain/Reward Loss": itemized["reward_loss"],
                    "Pretrain/L1 Loss": itemized["l1_loss"],
                    "Pretrain/L2 Loss": itemized["l2_loss"],
                    "Pretrain/TCN Accuracy": itemized["tcn_accuracy"],
                    "Pretrain/Reward Accuracy": itemized["reward_accuracy"],
                    "Pretrain/Learning Rate": lr,
                    "Pretrain/Step Time": step_time,
                },
            )

        elif self.model_arch == "v-cond":
            self.log(
                self.global_step,
                metrics={
                    "Pretrain/Step": self.global_step,
                    "Pretrain/Epoch": epoch,
                    "Pretrain/V-Cond Train Loss": loss,
                    "Pretrain/Reconstruction Loss": itemized["reconstruction_loss"],
                    "Pretrain/Learning Rate": lr,
                    "Pretrain/Step Time": step_time,
                },
            )

        elif self.model_arch == "v-dual":
            self.log(
                self.global_step,
                metrics={
                    "Pretrain/Step": self.global_step,
                    "Pretrain/Epoch": epoch,
                    "Pretrain/V-Dual Train Loss": loss,
                    "Pretrain/Reconstruction Loss": itemized["reconstruction_loss"],
                    "Pretrain/Zero Reconstruction Loss": itemized["zero_reconstruction_loss"],
                    "Pretrain/K Reconstruction Loss": itemized["k_reconstruction_loss"],
                    "Pretrain/Learning Rate": lr,
                    "Pretrain/Step Time": step_time,
                },
            )

        elif self.model_arch == "v-gen":
            self.log(
                self.global_step,
                metrics={
                    "Pretrain/Step": self.global_step,
                    "Pretrain/Epoch": epoch,
                    "Pretrain/V-Gen Train Loss": loss,
                    "Pretrain/Reconstruction Loss": itemized["reconstruction_loss"],
                    "Pretrain/Zero Reconstruction Loss": itemized["zero_reconstruction_loss"],
                    "Pretrain/K Reconstruction Loss": itemized["k_reconstruction_loss"],
                    "Pretrain/CLM Loss": itemized["lm_loss"],
                    "Pretrain/CLM Perplexity": itemized["lm_ppl"],
                    "Pretrain/LM Loss": itemized["lm_loss"],
                    "Pretrain/LM Perplexity": itemized["lm_ppl"],
                    "Pretrain/Learning Rate": lr,
                    "Pretrain/Step Time": step_time,
                },
            )

        else:
            raise ValueError(f"Metrics.push() for Model `{self.model_arch}` is not implemented!")

        return status

    def push_epoch(self, epoch: int, val_loss: torch.Tensor) -> Tuple[str, torch.Tensor, int]:
        """End-of-Epoch => Push accumulated metrics to loggers with model-specific handling."""
        if not self.is_rank_zero:
            return

        # Compute End-of-Epoch Specialized Metrics
        loss, step_time = torch.stack(list(self.tracker["loss"])).mean(), np.mean(list(self.tracker["step_time"]))
        lr, duration = self.tracker["lr"][-1], int(time.time() - self.start_time) + self.resume_time
        epoch_status = (
            f"[Epoch {epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f} -- Loss :: {loss:.4f} "
            f"-- Val Loss :: {val_loss:.4f} -- Total Time (sec) :: {duration}"
        )

        # Log for Model
        p_arch = {
            "v-mvp": "MVP",
            "v-r3m": "R3M (ViT)",
            "v-rn3m": "R3M (RN)",
            "v-cond": "V-Cond",
            "v-dual": "V-Dual",
            "v-gen": "V-Gen",
        }[self.model_arch]
        self.log(
            self.global_step,
            metrics={
                "Pretrain/Step": self.global_step,
                "Pretrain/Epoch": epoch,
                "Pretrain/Training Duration": duration,
                f"Pretrain/{p_arch} Train Epoch Loss": loss.item(),
                f"Pretrain/{p_arch} Train Loss": loss.item(),
                f"Pretrain/{p_arch} Validation Loss": val_loss.item(),
                "Pretrain/Learning Rate": lr,
                "Pretrain/Step Time": step_time,
            },
        )

        return epoch_status, loss, duration
