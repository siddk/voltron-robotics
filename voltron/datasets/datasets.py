"""
datasets.py

Core Pytorch Dataset implementations for the various "data flavors" used by the different representation learning models
(Voltron and data-locked reproductions). Crucially, ach dataset loads from the corresponding serialized batch files that
define the exact data (and order of iterating through the data) to see during each epoch.

Notably, these serialized files control exactly what data is seen by *all* methods *across epochs*; using these files is
critical to reproducibility & comparisons.

The file contains logic for a "standard" Dataset; all files (batch index files, image/video/language files) are stored
on local disk, assuming storage conducive to fast random reads. For a "streaming" dataset (loading data directly from
GCP Buckets/Amazon S3), see `v1/stream_datasets.py`.
"""
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose

from voltron.preprocessing.transforms import get_online_transform


class PretrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.epoch, self.h5, self.vid, self.states = 0, None, None, None
        self.index_path, self.language_path, self.language = None, None, None

    def hydrate(self, path: Path) -> None:
        # Create Open HDF5 Handle
        self.h5 = h5py.File(path, "r")
        self.vid, self.states = self.h5["vid"].asstr(), self.h5["states"].asstr()

        # Load Language Index
        if self.language_path is not None:
            self.language = torch.load(self.index_path / self.language_path)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError("PretrainDataset is an abstract class; should never be initialized directly!")

    def __len__(self) -> int:
        raise NotImplementedError("PretrainDataset is an abstract class; should never be initialized directly!")


class StateDataset(PretrainDataset):
    def __init__(self, epoch: int, index_path: Path, img_transform: Compose, is_val: bool = False) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None

        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / "state" / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / "state" / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return processed image frame as a Tensor."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        return self.img_transform(read_image(str(self.index_path.parent / self.states[idx][0])))

    def __len__(self) -> int:
        return self.n_examples


class StateLanguageDataset(PretrainDataset):
    def __init__(
        self,
        epoch: int,
        index_path: Path,
        img_transform: Compose,
        lang_dropout: Optional[float] = None,
        is_val: bool = False,
    ) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None
        self.lang_dropout, self.dropout_idxs = 0.0 if (lang_dropout is None) else lang_dropout, set()

        # Set Language Path
        self.language_path = "val-language-index.pt" if self.is_val else "train-language-index.pt"

        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / "state+language" / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / "state+language" / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

        # Assemble Dropout Indices
        n_drop = int(self.lang_dropout * self.n_examples)
        self.dropout_idxs = set(np.random.choice(self.n_examples, n_drop, replace=False))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return processed image frame and language, decomposed as the input_ids and attention_mask."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        # Get Vid ID --> parse out language, transform frame!
        vid = self.vid[idx]
        lang, lang_mask = self.language[vid]["input_ids"], self.language[vid]["attention_mask"]

        # Dropout Language (Naive Zeroing leads to NaN --> just want the "CLS" token)
        if idx in self.dropout_idxs:
            # Initial language token is *always* <CLS> = `101` --> last token always <SEP> = `102`
            lang[1:] *= 0
            lang_mask[1:] *= 0

        # Retrieve Frame & Return
        img = self.img_transform(read_image(str(self.index_path.parent / self.states[idx][0])))
        return img, lang, lang_mask

    def __len__(self) -> int:
        return self.n_examples


class StateOKDataset(PretrainDataset):
    def __init__(
        self,
        epoch: int,
        index_path: Path,
        img_transform: Compose,
        lang_dropout: Optional[float] = None,
        is_val: bool = False,
    ) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None
        self.lang_dropout, self.dropout_idxs = 0.0 if (lang_dropout is None) else lang_dropout, set()

        # Set Language Path
        self.language_path = "val-language-index.pt" if self.is_val else "train-language-index.pt"

        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / "state+ok" / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / "state+ok" / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

        # Assemble Dropout Indices
        n_drop = int(self.lang_dropout * self.n_examples)
        self.dropout_idxs = set(np.random.choice(self.n_examples, n_drop, replace=False))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return processed dual frames and language, decomposed as the input_ids and attention_mask."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        # Get Vid ID --> parse out language, transform frames!
        vid = self.vid[idx]
        lang, lang_mask = self.language[vid]["input_ids"], self.language[vid]["attention_mask"]

        # Dropout Language (Naive Zeroing leads to NaN --> just want the "CLS" token)
        if idx in self.dropout_idxs:
            # Initial language token is *always* <CLS> = `101` --> last token always <SEP> = `102`
            lang[1:] *= 0
            lang_mask[1:] *= 0

        # Retrieve Frames & Return
        imgs = self.states[idx]
        imgs = torch.stack([self.img_transform(read_image(str(self.index_path.parent / fn))) for fn in imgs])
        return imgs, lang, lang_mask

    def __len__(self) -> int:
        return self.n_examples


class GenStateOKDataset(PretrainDataset):
    def __init__(
        self,
        epoch: int,
        index_path: Path,
        img_transform: Compose,
        gen_ratio: float,
        is_val: bool = False,
    ) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None
        self.gen_ratio, self.gen_idxs = gen_ratio, set()

        # Set Language Path
        self.language_path = "val-language-index.pt" if self.is_val else "train-language-index.pt"

        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / "state+ok" / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / "state+ok" / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

        # Assemble Generation Indices
        n_gen = int(self.gen_ratio * self.n_examples)
        self.gen_idxs = set(np.random.choice(self.n_examples, n_gen, replace=False))

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Return dual frames, conditioning language, language to generate, decomposed as input_ids/attention mask."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        # Get Vid ID --> parse out language to condition on / generate
        vid = self.vid[idx]
        lang_con, lang_con_mask = self.language[vid]["input_ids"], self.language[vid]["attention_mask"]
        lang_gen, lang_gen_mask, lang_gen_weight = lang_con.clone(), lang_con_mask.clone(), None

        # Generate / Condition Check --> (Naive Zeroing leads to NaN --> just want the "CLS" token)
        if idx in self.gen_idxs:
            # When Generating --> just condition on the <CLS> token and generate the rest!
            lang_con[1:] *= 0
            lang_con_mask[1:] *= 0
            lang_gen_weight = 1

        else:
            # When Conditioning -> just generate the <CLS> token (so things don't crash) but set weight to 0
            lang_gen[1:] *= 0
            lang_gen_mask[1:] *= 0
            lang_gen_weight = 0

        # Retrieve Frames and Return
        imgs = self.states[idx]
        imgs = torch.stack([self.img_transform(read_image(str(self.index_path.parent / fn))) for fn in imgs])

        return imgs, lang_con, lang_con_mask, lang_gen, lang_gen_mask, lang_gen_weight

    def __len__(self) -> int:
        return self.n_examples


class QuintetDataset(PretrainDataset):
    def __init__(self, epoch: int, index_path: Path, img_transform: Compose, is_val: bool = False) -> None:
        super().__init__()
        self.index_path, self.is_val, self.val_loaded = index_path, is_val, False
        self.epoch, self.img_transform, self.hdf5_path, self.n_examples = epoch, img_transform, None, None

        # Set Language Path
        self.language_path = "val-language-index.pt" if self.is_val else "train-language-index.pt"

        # === Retrieve Epoch Batches --> only call before/between epochs (as we create new DataLoaders) ===
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Load Validation Batches
        if self.is_val and not self.val_loaded:
            self.hdf5_path = self.index_path / "quintet+language" / "validation-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

            # Set `val_loaded`
            self.val_loaded = True

        # Load Train Batches
        elif not self.is_val:
            self.hdf5_path = self.index_path / "quintet+language" / f"train-epoch={epoch}-batches.hdf5"
            with h5py.File(self.hdf5_path, "r") as h5:
                self.n_examples = len(h5["states"])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all five processed frames and language, decomposed as the input_ids/attention_mask."""
        if self.h5 is None:
            self.hydrate(self.hdf5_path)

        # Get Vid ID --> parse out language, transform frames!
        vid = self.vid[idx]
        lang, lang_mask = self.language[vid]["input_ids"], self.language[vid]["attention_mask"]

        # Retrieve Frames & Return
        imgs = self.states[idx]
        imgs = torch.stack([self.img_transform(read_image(str(self.index_path.parent / fn))) for fn in imgs])
        return imgs, lang, lang_mask

    def __len__(self) -> int:
        return self.n_examples


def get_datasets(
    epoch: int,
    dataset_name: str,
    model_arch: str,
    artifact_path: str,
    data_modality: str,
    resolution: int,
    normalization: Tuple[Any, Any],
    lang_dropout: Optional[float] = None,
    gen_ratio: Optional[float] = None,
) -> Tuple[PretrainDataset, PretrainDataset]:
    index = Path(artifact_path) / dataset_name / "index"
    img_transform = get_online_transform(dataset_name, model_arch, resolution, normalization)

    # Switch on `data_modality` --> differs based on `model_arch` (e.g., MVP --> img, V-Cond --> img, language)
    if data_modality == "state":
        train_ds = StateDataset(epoch, index, img_transform)
        val_ds = StateDataset(epoch, index, img_transform, is_val=True)

    elif data_modality == "state+language":
        train_ds = StateLanguageDataset(epoch, index, img_transform, lang_dropout)
        val_ds = StateLanguageDataset(epoch, index, img_transform, lang_dropout, is_val=True)

    elif data_modality == "state+ok":
        # V-Dual --> don't return language modeling elements (causal attention mask, suffix language, etc.)
        if gen_ratio is None:
            train_ds = StateOKDataset(epoch, index, img_transform, lang_dropout)
            val_ds = StateOKDataset(epoch, index, img_transform, lang_dropout, is_val=True)

        # V-Gen --> add language modeling elements!
        else:
            train_ds = GenStateOKDataset(epoch, index, img_transform, gen_ratio)
            val_ds = GenStateOKDataset(epoch, index, img_transform, gen_ratio, is_val=True)

    elif data_modality == "quintet+language":
        train_ds = QuintetDataset(epoch, index, img_transform)
        val_ds = QuintetDataset(epoch, index, img_transform, is_val=True)

    else:
        raise ValueError(f"Data Modality `{data_modality}` is not supported!")

    return train_ds, val_ds
