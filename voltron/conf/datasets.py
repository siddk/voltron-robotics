"""
datasets.py

Base Hydra Structured Config for defining various pretraining datasets and appropriate configurations. Uses a simple,
single inheritance structure.
"""
from dataclasses import dataclass
from typing import Any, Tuple

from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    path: str = MISSING
    artifact_path: str = MISSING

    # Streaming Parameters (assumes fully preprocessed dataset lives at `stream_prefix/...`)
    #   =>> Deprecated as of `v2`
    stream: bool = True
    stream_prefix: str = "data/processed"

    # Dataset-Specific Parameters
    resolution: int = 224
    normalization: Tuple[Any, Any] = MISSING

    # For preprocessing --> maximum size of saved frames (assumed square)
    preprocess_resolution: int = MISSING

    # Validation Parameters
    n_val_videos: int = MISSING

    # Language Modeling Parameters
    language_model: str = "distilbert-base-uncased"
    hf_cache: str = to_absolute_path("data/hf-cache")

    # Maximum Length for truncating language inputs... should be computed after the fact (set to -1 to compute!)
    max_lang_len: int = MISSING

    # Dataset sets the number of pretraining epochs (general rule :: warmup should be ~5% of full)
    warmup_epochs: int = MISSING
    max_epochs: int = MISSING

    # Plausible Formats --> These are instantiations each "batch" could take, with a small DSL
    #   > Note: Assumes final element of the list is the "most expressive" --> used to back-off
    batch_formats: Any = (
        ("state", ("state_i",)),
        ("state+language", ("state_i", "language")),
        ("state+ok", ("state_initial", "state_i", "language")),
        ("quintet+language", ("state_initial", "state_i", "state_j", "state_k", "state_final", "language")),
    )

    # Preprocessing :: Frame-Sampling Parameters
    initial_final_alpha: float = 0.2


@dataclass
class SthSthv2Config(DatasetConfig):
    # fmt: off
    name: str = "sth-sth-v2"
    path: str = to_absolute_path("data/raw/sth-sth-v2")
    artifact_path: str = to_absolute_path("data/processed/sth-sth-v2")

    # Dataset Specific arguments
    normalization: Tuple[Any, Any] = (                              # Mean & Standard Deviation (default :: ImageNet)
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )

    # Sth-Sth-v2 Videos have a fixed height of 240; we'll crop to square at this resolution!
    preprocess_resolution: int = 240

    # Validation Parameters
    n_val_videos: int = 1000                                        # Number of Validation Clips (fast evaluation!)

    # Epochs for Dataset
    warmup_epochs: int = 20
    max_epochs: int = 400

    # Language Modeling Parameters
    max_lang_len: int = 20
    # fmt: on


# Create a configuration group `dataset` and populate with the above...
#   =>> Note :: this is meant to be extendable --> add arbitrary datasets & mixtures!
cs = ConfigStore.instance()
cs.store(group="dataset", name="sth-sth-v2", node=SthSthv2Config)
