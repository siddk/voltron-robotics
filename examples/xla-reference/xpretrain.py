"""
xpretrain.py

(The `x` prefix indicates this is a script geared for XLA/TPU backends *only*)!

Reference script for PyTorch XLA (TPU-based) pretraining on the non-Qualcomm version of Sth-Sth-v2; this is
mostly for completeness =>> the hope is that the regular `pretrain.py` script is more general and maintained.

Focuses on multi-TPU (XLA) training --> but also supports single-core TPU training, as the default distributed mp.spawn
behavior just collapses into a single thread! Loads and preprocesses dataset, instantiates a model, and runs training.

Run with `python xtrain.py` (will by default use the configuration specified by `DEFAULTS` below).
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch_xla.core.xla_model as xm
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from voltron.conf import AcceleratorConfig, DatasetConfig, ModelConfig, TrackingConfig
from voltron.overwatch import OverwatchRich

# Set Defaults (Hydra w/ Structured Configs)
DEFAULTS = [
    "_self_",
    {"model": "v-cond"},
    {"dataset": "sth-sth-v2"},
    {"accelerator": "tpu-v3-8"},
    {"tracking": "voltron-tracking"},
    {"override hydra/job_logging": "overwatch_rich"},
]


@dataclass
class PretrainConfig:
    # fmt: off
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: Dict[str, Any] = field(default_factory=lambda: {
        "run": {"dir": "./runs/train/${model.identifier}+dataset-${dataset.name}"}
    })

    # Command Line Arguments
    run_id: Optional[str] = None                                        # Run ID for Logging
    seed: int = 21                                                      # Random Seed (for reproducibility)

    # Resume / Debug Behavior
    resume: bool = True                                                 # Whether to resume an existing run...
    resume_epoch: Optional[int] = None                                  # Epoch to resume (if auto-resuming)...
    checkpoint_path: Optional[str] = None                               # Path to the specific checkpoint to load!
    wandb_resume_id: Optional[str] = None                               # W&B Run ID for `resume` behavior...

    # Composable / Structured Arguments
    model: ModelConfig = MISSING                                        # Model architecture for pretraining
    dataset: DatasetConfig = MISSING                                    # List of datasets for pretraining
    accelerator: AcceleratorConfig = MISSING                            # Accelerator configuration
    tracking: TrackingConfig = MISSING                                  # Run/experiment tracking configuration
    # fmt: on


# Hydra Setup :: Retrieve ConfigStore (Singleton) & Register Components
cs = ConfigStore.instance()
cs.store(group="hydra/job_logging", name="overwatch_rich", node=OverwatchRich)  # Annoying - configure logger for Hydra
cs.store(name="config", node=PretrainConfig)


def xpretrain(cfg: PretrainConfig) -> None:
    # Identify if `is_rank_zero` --> We only log from the rank zero process!
    is_rank_zero = xm.is_master_ordinal(local=False)
    xm.master_print("Voltron Training :: Assembling the Legendary Defender...")

    # Create Unique Run Name -- if `resume = True` we assume same "run_id"
    run_id = cfg.run_id
    if run_id is None:
        run_id = run_dir = f"{cfg.model.identifier}+{cfg.dataset.name}-x{cfg.seed}"
        cfg.run_id = run_id
    else:
        cfg.run_id = run_dir = run_id

    if is_rank_zero:
        os.makedirs(run_dir, exist_ok=True)

    xm.master_print(
        '\t=>> "If you get too worried about what could go wrong, you might miss a chance to do something great."'
    )

    # Set Randomness, get DataLoader worker initialization function (to ensure any random augmentations!)
    # worker_init_fn = set_global_seed(cfg.seed)

    # Model Initialization Logic
    xm.master_print("Initializing Model and Placing on Different Devices...")
    if cfg.model.arch == "v-mvp":
        xm.master_print(f"Initializing MVP variant `{cfg.model.identifier}`")


def mp_fn(_: int, cfg: PretrainConfig) -> None:
    torch.set_default_tensor_type("torch.FloatTensor")

    # Let's Start Pretraining!
    xpretrain(cfg)


@hydra.main(config_path=None, config_name="config")
def main(cfg: PretrainConfig) -> None:
    import torch_xla.distributed.xla_multiprocessing as xmp

    # Call XMP Spawn w/ the Config as the sole argument...
    xmp.spawn(mp_fn, args=(cfg,), nprocs=cfg.accelerator.num_accelerators, start_method="spawn")


if __name__ == "__main__":
    main()
