"""
tracking.py

Base Hydra Structured Config for defining various run & experiment tracking configurations, e.g., via Weights & Biases.
Uses a simple single inheritance structure.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrackingConfig:
    # Active Loggers --> List of Loggers
    active_loggers: List[str] = field(default_factory=lambda: ["jsonl", "wandb"])

    # Generic Logging Frequency --> Matters more for XLA/TPUs... set this to be as large as you can stomach!
    log_frequency: int = 100

    # Checkpointing Strategy --> Save each epoch, keep most recent `idx[0]` checkpoints & *every* `idx[1]` checkpoints
    #   Additionally, save (locally) a checkpoint every `idx[2]` steps for the current epoch (-1).
    checkpoint_strategy: Tuple[int, int, int] = (1, 1, 1500)

    # Weights & Biases Setup
    project: str = "voltron-pretraining"
    entity: str = "voltron-robotics"

    # Notes & Tags are at the discretion of the user... see below
    notes: str = MISSING
    tags: Optional[List[str]] = None

    # Directory to save W&B Metadata & Logs in General -- if None, defaults to `logs/` in the Hydra CWD
    directory: Optional[str] = None


@dataclass
class VoltronTrackingConfig(TrackingConfig):
    # Note: I really like using notes to keep track of things, so will crash unless specified with run.
    #   > For `tags` I like to populate based on other args in the script, so letting it remain None
    notes: str = MISSING


# Create a configuration group `trackers` and populate with the above...
cs = ConfigStore.instance()
cs.store(group="tracking", name="voltron-tracking", node=VoltronTrackingConfig)
