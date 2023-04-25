"""
overwatch.py

Utility class for creating a centralized/standardized logger (to pass to Hydra), with a sane default format.
"""
from dataclasses import dataclass, field
from typing import Any, Dict

# Overwatch Default Format String
FORMATTER, DATEFMT = "[*] %(asctime)s - %(name)s >> %(levelname)s :: %(message)s", "%m/%d [%H:%M:%S]"
RICH_FORMATTER = "| >> %(message)s"


# Rich Overwatch Variant --> Good for debugging, and tracing!
@dataclass
class OverwatchRich:
    version: int = 1
    formatters: Dict[str, Any] = field(
        default_factory=lambda: {
            "simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT},
            "simple-file": {"format": FORMATTER, "datefmt": DATEFMT},
        }
    )
    handlers: Dict[str, Any] = field(
        default_factory=lambda: {
            "console": {
                "class": "rich.logging.RichHandler",
                "formatter": "simple-console",
                "rich_tracebacks": True,
                "show_level": True,
                "show_path": True,
                "show_time": True,
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "simple-file",
                "filename": "${hydra.job.name}.log",
            },
        }
    )
    root: Dict[str, Any] = field(default_factory=lambda: {"level": "INFO", "handlers": ["console", "file"]})
    disable_existing_loggers: bool = True


# Standard Overwatch Variant --> Performant, no bells & whistles
@dataclass
class OverwatchStandard:
    version: int = 1
    formatters: Dict[str, Any] = field(default_factory=lambda: {"simple": {"format": FORMATTER, "datefmt": DATEFMT}})
    handlers: Dict[str, Any] = field(
        default_factory=lambda: {
            "console": {"class": "logging.StreamHandler", "formatter": "simple", "stream": "ext://sys.stdout"},
            "file": {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": "${hydra.job.name}.log",
            },
        }
    )
    root: Dict[str, Any] = field(default_factory=lambda: {"level": "INFO", "handlers": ["console", "file"]})
    disable_existing_loggers: bool = True
