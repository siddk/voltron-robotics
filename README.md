# Voltron

> *Voltron*: Language-Driven Representation Learning for Robotics – Python package with code for loading pretrained models & pretraining Voltron, R3M, and MVP from scratch.

Package repository for Voltron: Language-Driven Representation Learning for Robotics.

Built with [PyTorch](https://pytorch.org/), using sane quality defaults (`black`, `ruff`, `pre-commit`).

---

## Installation

This repository is built on top of PyTorch; while specified as a dependency for the package, we highly recommend that
you install the desired version of PyTorch (e.g., with accelerator support) for your given hardware and dependency
manager (e.g., `conda`). Otherwise, the default installed version be incompatible.

PyTorch installation instructions [can be found here](https://pytorch.org/get-started/locally/). This repository
requires PyTorch >= 1.12, but has only been thoroughly tested with PyTorch 1.12.0, Torchvision 0.13.0, Torchaudio 0.12.0.

Once PyTorch has been properly installed, you can install this package locally via an editable installation:

```bash
git clone https://github.com/siddk/voltron-robotics
cd voltron-robotics
pip install -e .
```

## Usage

Project-specific usage notes...

## Contributing

Before committing to the repository, *make sure to set up your dev environment!*

Here are the basic development environment setup guidelines:

+ Fork/clone the repository, performing an editable installation. Make sure to install with the development dependencies
  (e.g., `pip install -e ".[dev]"`); this will install `black`, `ruff`, and `pre-commit`.

+ Install `pre-commit` hooks (`pre-commit install`).

+ Branch for the specific feature/issue, issuing PR against the upstream repository for review.

Additional Contribution Notes:
- This project has migrated to the recommended
  [`pyproject.toml` based configuration for setuptools](https://setuptools.pypa.io/en/latest/userguide/quickstart.html).
  However, given that several tools have not fully adopted [PEP 660](https://peps.python.org/pep-0660/), we provide a
  [`setup.py` file for backwards compatibility](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html).

- This package follows the [`flat-layout` structure](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#flat-layout)
  described in `setuptools`.

- Make sure to add any new dependencies to the `project.toml` file!

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `docs/` - Package documentation - including project roadmap, additional notes (if any).
+ `voltron` - Package source code; has all core utilities for model specification, loading,
                               preprocessing, etc.
+ `scripts/` - Standalone scripts for various functionality (e.g., training).
+ `.gitignore` - Default Python `.gitignore`.
+ `.pre-commit-config.yaml` - Pre-commit configuration file (sane defaults + `black` + `ruff`).
+ `LICENSE` - By default, research code is made available under the MIT License; if changing, think carefully about why!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Following PEP 621, this file has all project configuration details (including dependencies), as
                     well as tool configurations (for `black` and `ruff`).
+ `README.md` - You are here!
