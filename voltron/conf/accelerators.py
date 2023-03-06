"""
accelerator.py

Base Hydra Structured Configs for defining various accelerator schemes. Uses a simple single inheritance structure.
"""
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class AcceleratorConfig:
    accelerator: str = MISSING
    num_accelerators: int = MISSING
    num_workers: int = MISSING


@dataclass
class TPUv2OneConfig(AcceleratorConfig):
    accelerator = "tpu"
    num_accelerators = 1
    num_workers = 4


@dataclass
class TPUv2EightConfig(AcceleratorConfig):
    accelerator = "tpu"
    num_accelerators = 8
    num_workers = 4


@dataclass
class TPUv3OneConfig(AcceleratorConfig):
    accelerator = "tpu"
    num_accelerators = 1
    num_workers = 8


@dataclass
class TPUv3EightConfig(AcceleratorConfig):
    accelerator = "tpu"
    num_accelerators = 8
    num_workers = 8


# Create a configuration group `accelerator` and populate with the above...
cs = ConfigStore.instance()
cs.store(group="accelerator", name="tpu-v2-1", node=TPUv2OneConfig)
cs.store(group="accelerator", name="tpu-v2-8", node=TPUv2EightConfig)
cs.store(group="accelerator", name="tpu-v3-1", node=TPUv3OneConfig)
cs.store(group="accelerator", name="tpu-v3-8", node=TPUv3EightConfig)
