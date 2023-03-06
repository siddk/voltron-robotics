# XLA Reference

We trained the original Voltron models (and data-locked reproductions of R3M and MVP) on TPU v3-8 nodes generously
provided by the [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program. At the time we started
the project, PyTorch XLA still had some bumps, which was further complicated by the switch from
[TPU Nodes to TPU VMs](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-arch).

To get things to work, we had to add some non-intuitive code to facilitate PyTorch + TPUs (vs. a standard distributed
data parallel training pipeline). As a result, `xpretrain.py` is here mostly for documentation purposes, with a fully
refactored version `pretrain.py` forthcoming.

We also include the original cloud preprocesssing script `xpreprocess.py` for completeness.
