# Project Roadmap

We document the future of this project (new features to be added, issues to address) here. For the most part, any
new features/bugfixes are documented as [Github Issues](https://github.com/siddk/voltron-robotics/issues).

## Timeline

[X] - **February 26th, 2023**: Initial Voltron-Robotics release with support for loading/adapting all pretrained models,
                               with comprehensive verification scripts & a small adaptation example.

[X] - **April 4, 2023**:  [#1](https://github.com/siddk/voltron-robotics/issues/1) - Add `xpretrain.py` reference script,
                          mostly for completeness. Refactor/rewrite the preprocessing and pretraining pipeline to reflect
                          the Qualcomm Sth-Sth data format, as well as PyTorch DDP vs. the patched PyTorch XLA!

[X] - **April 11, 2023**: [#2](https://github.com/siddk/voltron-robotics/issues/2) - Add support and a more general API
                          for pretraining on other datasets.

[ ] - **Future**:         [#5](https://github.com/siddk/voltron-robotics/issues/5) - Add better documentation and examples
                          around using the MAP extractor (especially for adaptation tasks).
