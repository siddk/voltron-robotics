"""
xpreprocess.py

Centralized script for preprocessing Sth-Sth-v2 for TPU/GCP pretraining, using a multi-stage, multiprocessing strategy.

Mean to run as a standalone script, *prior* to calling `xpretrain.py` =>> mostly because we want to preprocess the data
once, as a fixed cost.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from voltron.conf import DatasetConfig
from voltron.overwatch import OverwatchRich
from voltron.preprocessing import index, jsonify_language, preprocess_language, preprocess_videos, unify_batches
from voltron.util import set_global_seed

# Grab Logger
overwatch = logging.getLogger(__file__)


# Set Defaults (Hydra w/ Structured Configs)
DEFAULTS = ["_self_", {"dataset": "sth-sth-v2"}, {"override hydra/job_logging": "overwatch_rich"}]


@dataclass
class PreprocessingConfig:
    # fmt: off
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: Dict[str, Any] = field(
        default_factory=lambda: {"run": {"dir": "./runs/preprocessing/${now:%m-%d}/dataset-${dataset.name}"}}
    )

    # Command Line Arguments
    seed: int = 21                                  # Random Seed (for reproducibility)
    dry_run: bool = False                           # Dry Run --> Get a sense of preprocessing/serialization footprint

    # Composable / Structured Arguments
    dataset: DatasetConfig = MISSING                # Dataset(s) for pretraining/preprocessing
    # fmt: on


# Hydra Setup :: Retrieve ConfigStore (Singleton) & Register Components
cs = ConfigStore.instance()
cs.store(group="hydra/job_logging", name="overwatch_rich", node=OverwatchRich)  # Annoying - configure logger for Hydra
cs.store(name="config", node=PreprocessingConfig)


@hydra.main(config_path=None, config_name="config")
def xpreprocess(cfg: PreprocessingConfig) -> None:
    overwatch.info("Preprocessing :: Running Phases for Frame Extraction, Language Compilation, and Batching...")

    # Set Randomness
    set_global_seed(cfg.seed)

    # Phase 1 :: Serialize Frames from Video Clips --> Get `registry` for train and val (index structure)
    train_registry, val_registry, train_dir, val_dir = preprocess_videos(
        cfg.dataset.name,
        path=cfg.dataset.path,
        artifact_path=cfg.dataset.artifact_path,
        resolution=cfg.dataset.resolution,
        n_val_videos=cfg.dataset.n_val_videos,
        dry_run=cfg.dry_run,
    )

    # Phase 2 :: Normalize & Tokenize Language  --> Create `index.pt` & `index.json` files
    preprocess_language(
        cfg.dataset.name,
        train_registry,
        val_registry,
        max_lang_len=cfg.dataset.max_lang_len,
        language_model=cfg.dataset.language_model,
        hf_cache=cfg.dataset.hf_cache,
    )
    jsonify_language(train_registry, val_registry)
    index_dir = index(train_registry, val_registry, cfg.dataset.name, artifact_path=cfg.dataset.artifact_path)

    # Phase 3 :: Assemble & Unify Batch "Sets" across the Varied Dataset Formats (for each Model =>> "data-locked")
    unify_batches(
        cfg.dataset.artifact_path,
        cfg.dataset.name,
        train_registry,
        val_registry,
        train_dir,
        val_dir,
        index_dir,
        cfg.dataset.batch_formats,
        max_epochs=cfg.dataset.max_epochs,
        initial_final_alpha=cfg.dataset.initial_final_alpha,
    )


if __name__ == "__main__":
    xpreprocess()
