"""
stream_datasets.py

Core PyTorch Datasets for the various "flavors" of data used by the various models under study. Crucially, each dataset
loads from the corresponding "batch" serialized files, that define the exact data to use.

Notably, these serialized files control exactly what data is seen by *all* methods **across epochs.** Using them is
fairly critical to reproducibility & fair comparison.

This specific file contains logic for a "streaming" Dataset; data is fetched (within the dataloader, by each
worker) via an open connection over the network to a GCS bucket, materializing data as raw BytesIO objects fed to
PIL.Image constructors.
"""
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from google.api_core.exceptions import NotFound
from google.auth.exceptions import TransportError
from google.cloud import storage
from google.resumable_media._helpers import _LOGGER
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, get_worker_info
from torchvision.io import read_image
from torchvision.transforms import Compose
from torchvision.transforms.functional import pil_to_tensor

from voltron.preprocessing.v1.transforms import get_online_transform
from voltron.util.distributed import get_rank

# NOTE --> IF STREAMING JPEGS, WE NEED TO USE PILLOW TO READ FILES (w/o extracting locally...)
#   =>> Instead of `read_image(file)` assume we have "fname" and open fileobj (as BytesIO) -- remember to `seek(0)`
#
# > from PIL import Image
# > from torchvision.transforms.functional import pil_to_tensor
# > tensor = pil_to_tensor(Image.open(fileobj)
#       |--> This returns a `torch.uint8` Tensor of shape [3, 224, 224] --> *verified* equivalent to `read_image`

# Create Global GCS Client...
#   =>> Multiprocessing.spawn() does not inherit from base shell --> need to set service account key...
#   =>> TODO :: Figure out how to fetch num_accelerators & num_workers programatically...
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/mnt/home/auth/gcp-auth.json"
N_CORES, BUCKETS = 8, [storage.Client().bucket("voltron-ANONYMIZED") for _ in range(8 * 8)]

# Suppress Google Cloud Loggers
_LOGGER.propagate = False
storage.blob._logger.propagate = False


class PretrainDataset(Dataset):
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class StateDataset(PretrainDataset):
    def __init__(
        self,
        epoch: int,
        index_path: Path,
        img_transform: Compose,
        stream: bool = False,
        prefix: Optional[Path] = None,
        is_val: bool = False,
        do_retry: bool = True,
        n_retries: int = 3,
    ) -> None:
        super().__init__()
        self.index_path, self.stream, self.is_val, self.val_loaded = index_path, stream, is_val, False
        self.epoch, self.transform, self.elements, self.prefix = epoch, img_transform, None, prefix
        self.r = N_CORES * get_rank()
        self.do_retry, self.n_retries = do_retry, n_retries

        # === Retrieve Epoch Batches (only call `set_epoch` inside __init__() ===
        self.set_epoch(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        # Not Streaming --> Read from local disk...
        if not self.stream:
            if self.is_val and not self.val_loaded:
                with open(self.index_path / "state" / "validation-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                with open(self.index_path / "state" / f"train-epoch={epoch}-batches.json", "r") as f:
                    self.elements = json.load(f)

        # Streaming --> Beam directly from Bucket
        else:
            if self.is_val and not self.val_loaded:
                blob = BUCKETS[self.r].blob(str(self.prefix / "index" / "state" / "validation-batches.json"))
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["state"] currently maps to disk path... remove all but `parent/child.jpg`
                for element in self.elements:
                    element["state"] = "/".join(element["state"].split("/")[-2:])

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                blob = BUCKETS[self.r].blob(str(self.prefix / "index" / "state" / f"train-epoch={epoch}-batches.json"))
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["state"]` currently maps to disk path... remove all but `parent/child.jpg`
                for element in self.elements:
                    element["state"] = "/".join(element["state"].split("/")[-2:])

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return single frame as torch Tensor."""
        if not self.stream:
            return self.transform(read_image(self.elements[index]["state"]))
        else:
            # Multiplex w/ num_worker idx...
            worker_info = get_worker_info()
            r = (self.r + worker_info.id) if worker_info is not None else self.r

            # Streaming + Retry Logic (in case of a bad connection -- retry same file!)
            frame_path = self.elements[index]["state"]
            for _i in range(self.n_retries):
                try:
                    # Stream JPEG contents into BytesIO (seek back to 0), then into PIL Image.open()
                    if self.is_val:
                        blob, fobj = BUCKETS[r].blob(str(self.prefix / "val" / frame_path)), BytesIO()
                    else:
                        blob, fobj = BUCKETS[r].blob(str(self.prefix / "train" / frame_path)), BytesIO()

                    # Error Handling --> we've already run several dry-run/verification trials, but about ~0.004% of
                    #                    the time, we'll hit some sort of TCP/Transport error; this might even go up
                    #                    with multiple runs happening at the same time.
                    #
                    #                    To address this, we're adopting the simplest possible "retry" strategy that
                    #                    immediately tries to re-download the same file (and crashes if not possible).
                    #                    This ensures reproducibility, but puts some extra effort onto the user...

                    # File download...
                    blob.download_to_file(fobj)
                    fobj.seek(0)

                    # Image loading...
                    img_tensor = pil_to_tensor(Image.open(fobj))

                    # Return transformed image...
                    return self.transform(img_tensor)

                except (NotFound, TransportError, UnidentifiedImageError, OSError) as e:
                    # At the minimum --> print the broken file (obnoxiously!)
                    print(f"=>> BROKEN FILE :: {frame_path}")
                    if not self.do_retry:
                        raise e
                    else:
                        continue

            # If we've exhausted our retries --> raise an informative ValueError
            raise ValueError(f"Failed to fix state `{self.elements[index]['state']}` w/ {self.n_retries} retries...")

    def __len__(self) -> int:
        return len(self.elements)


class StateLanguageDataset(PretrainDataset):
    def __init__(
        self,
        epoch: int,
        index_path: Path,
        img_transform: Compose,
        lang_dropout: Optional[float] = None,
        stream: bool = False,
        prefix: Optional[Path] = None,
        is_val: bool = False,
        do_retry: bool = True,
        n_retries: int = 3,
    ) -> None:
        super().__init__()
        self.index_path, self.stream, self.is_val, self.val_loaded = index_path, stream, is_val, False
        self.epoch, self.transform, self.elements, self.prefix = epoch, img_transform, None, prefix
        self.lang_dropout = 0.0 if (lang_dropout is None or lang_dropout == 0) else lang_dropout
        self.dropout_indices = set()
        self.r = N_CORES * get_rank()
        self.do_retry, self.n_retries = do_retry, n_retries

        # Load Language Index & Retrieve Epoch 0 Batches
        language_path = "val-language-index.json" if self.is_val else "train-language-index.json"
        if not self.stream:
            with open(self.index_path / language_path, "r") as f:
                self.language_index = json.load(f)
        else:
            blob = BUCKETS[self.r].blob(str(self.prefix / "index" / language_path))
            self.language_index = json.loads(blob.download_as_string())

        # === Retrieve Epoch Batches (only call `set_epoch` inside __init__() ===
        self.set_epoch(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        # Not Streaming --> Read from local disk...
        if not self.stream:
            if self.is_val and not self.val_loaded:
                with open(self.index_path / "state+language" / "validation-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                with open(self.index_path / "state+language" / f"train-epoch={epoch}-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

        # Streaming --> Beam directly from Bucket
        else:
            if self.is_val and not self.val_loaded:
                blob = BUCKETS[self.r].blob(str(self.prefix / "index" / "state+language" / "validation-batches.json"))
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["state"]` currently maps to disk path... remove all but `parent/child.jpg`
                for element in self.elements:
                    element["state"] = "/".join(element["state"].split("/")[-2:])

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                blob = BUCKETS[self.r].blob(
                    str(self.prefix / "index" / "state+language" / f"train-epoch={epoch}-batches.json")
                )
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["state"] currently maps to disk path... remove all but `parent/child.jpg`
                for element in self.elements:
                    element["state"] = "/".join(element["state"].split("/")[-2:])

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the frame and language, decomposed as the input_ids, and attention_mask."""
        vid = self.elements[index]["vid"]
        lang = torch.tensor(self.language_index[vid]["input_ids"], dtype=torch.int64)
        lang_mask = torch.tensor(self.language_index[vid]["attention_mask"], dtype=torch.int64)

        # Dropout Language Check --> (Naive Zeroing leads to NaN --> just want the "CLS" token)
        if index in self.dropout_indices:
            # Initial language token is *always* <CLS> = `101` --> last token always <SEP> = `102`
            lang[1:] *= 0
            lang_mask[1:] *= 0

        # Retrieve Single Frame
        if not self.stream:
            img = self.transform(read_image(self.elements[index]["state"]))
            return img, lang, lang_mask
        else:
            # Multiplex w/ num_worker idx...
            worker_info = get_worker_info()
            r = (self.r + worker_info.id) if worker_info is not None else self.r

            # Streaming + Retry Logic (in case of a bad connection -- retry same file!)
            frame_path = self.elements[index]["state"]
            for _i in range(self.n_retries):
                try:
                    # Stream JPEG contents into BytesIO (seek back to 0), then into PIL Image.open()
                    if self.is_val:
                        blob, fobj = BUCKETS[r].blob(str(self.prefix / "val" / frame_path)), BytesIO()
                    else:
                        blob, fobj = BUCKETS[r].blob(str(self.prefix / "train" / frame_path)), BytesIO()

                    # Error Handling --> we've already run several dry-run/verification trials, but about ~0.004% of
                    #                    the time, we'll hit some sort of TCP/Transport error; this might even go up
                    #                    with multiple runs happening at the same time.
                    #
                    #                    To address this, we're adopting the simplest possible "retry" strategy that
                    #                    immediately tries to re-download the same file (and crashes if not possible).
                    #                    This ensures reproducibility, but puts some extra effort onto the user...

                    # File download...
                    blob.download_to_file(fobj)
                    fobj.seek(0)

                    # Image loading...
                    img_tensor = pil_to_tensor(Image.open(fobj))

                    # Assemble transformed image and return...
                    img = self.transform(img_tensor)
                    return img, lang, lang_mask

                except (NotFound, TransportError, UnidentifiedImageError, OSError) as e:
                    # At the minimum --> print the broken file (obnoxiously!)
                    print(f"=>> BROKEN FILE :: {frame_path}")
                    if not self.do_retry:
                        raise e
                    else:
                        continue

            # If we've exhausted our retries --> raise an informative ValueError
            raise ValueError(f"Failed to fix state `{self.elements[index]['state']}` w/ {self.n_retries} retries...")

    def __len__(self) -> int:
        return len(self.elements)


class StateOKDataset(PretrainDataset):
    def __init__(
        self,
        epoch: int,
        index_path: Path,
        img_transform: Compose,
        lang_dropout: Optional[float] = None,
        stream: bool = False,
        prefix: Optional[Path] = None,
        no_lang: bool = False,
        is_val: bool = False,
        do_retry: bool = True,
        n_retries: int = 3,
    ) -> None:
        super().__init__()
        self.index_path, self.stream, self.is_val, self.val_loaded = index_path, stream, is_val, False
        self.epoch, self.transform, self.elements, self.prefix = epoch, img_transform, None, prefix
        self.no_lang, self.lang_dropout = no_lang, 0.0 if (lang_dropout is None or lang_dropout == 0) else lang_dropout
        self.dropout_indices = set()
        self.r = N_CORES * get_rank()
        self.do_retry, self.n_retries = do_retry, n_retries

        # Load Language Index & Retrieve Epoch 0 Batches
        if not self.no_lang:
            language_path = "val-language-index.json" if self.is_val else "train-language-index.json"
            if not self.stream:
                with open(self.index_path / language_path, "r") as f:
                    self.language_index = json.load(f)
            else:
                blob = BUCKETS[self.r].blob(str(self.prefix / "index" / language_path))
                self.language_index = json.loads(blob.download_as_string())

        # === Retrieve Epoch Batches (only call `set_epoch` inside __init__() ===
        self.set_epoch(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        # Not Streaming --> Read from local disk...
        if not self.stream:
            if self.is_val and not self.val_loaded:
                with open(self.index_path / "state+ok" / "validation-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                with open(self.index_path / "state + ok" / f"train-epoch={epoch}-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

        # Streaming --> Beam directly from Bucket
        else:
            if self.is_val and not self.val_loaded:
                blob = BUCKETS[self.r].blob(str(self.prefix / "index" / "state+ok" / "validation-batches.json"))
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["states"]` currently maps to disk path.. remove all but `parent/child.jpg`
                for element in self.elements:
                    element["states"] = ["/".join(x.split("/")[-2:]) for x in element["states"]]

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                blob = BUCKETS[self.r].blob(
                    str(self.prefix / "index" / "state+ok" / f"train-epoch={epoch}-batches.json")
                )
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["states"]` currently maps to disk path.. remove all but `parent/child.jpg`
                for element in self.elements:
                    element["states"] = ["/".join(x.split("/")[-2:]) for x in element["states"]]

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

    # ruff: noqa: C901
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return both states/frames and language, decomposed as the input_ids and attention_mask."""
        vid = self.elements[index]["vid"]

        # Fetch language if desired...
        if not self.no_lang:
            lang = torch.tensor(self.language_index[vid]["input_ids"], dtype=torch.int64)
            lang_mask = torch.tensor(self.language_index[vid]["attention_mask"], dtype=torch.int64)

            # Dropout Language Check --> (Naive Zeroing leads to NaN --> just want the "CLS" token)
            if index in self.dropout_indices:
                # Initial language token is *always* <CLS> = `101` --> last token always <SEP> = `102`
                lang[1:] *= 0
                lang_mask[1:] *= 0

        # Retrieve Frames
        if not self.stream:
            imgs = self.elements[index]["states"]
            imgs = torch.stack([self.transform(read_image(s)) for s in imgs])

            # Return --> based on `self.no_lang`
            if not self.no_lang:
                return imgs, lang, lang_mask
            else:
                return imgs

        else:
            # Multiplex w/ num_worker idx...
            worker_info = get_worker_info()
            r = (self.r + worker_info.id) if worker_info is not None else self.r

            # Streaming + Retry Logic (in case of a bad connection -- retry same files!)
            frame_paths, current_frame = list(self.elements[index]["states"]), None
            for _i in range(self.n_retries):
                try:
                    # Stream JPEG contents into BytesIO (seek back to 0), then PIL Image.open()
                    imgs = []
                    for _current_idx, current_frame in enumerate(frame_paths):
                        if self.is_val:
                            blob, fobj = BUCKETS[r].blob(str(self.prefix / "val" / current_frame)), BytesIO()
                        else:
                            blob, fobj = BUCKETS[r].blob(str(self.prefix / "train" / current_frame)), BytesIO()

                        # Error Handling --> we've already run several dry-run/verification trials, but about ~0.004% of
                        #                    the time, we'll hit some sort of TCP/Transport error; this might even go up
                        #                    with multiple runs happening at the same time.
                        #
                        #                    To address this, we're adopting the simplest possible "retry" strategy that
                        #                    immediately tries to re-download the same file (crashes if not possible).
                        #                    This ensures reproducibility, but puts some extra effort onto the user...

                        # File download...
                        blob.download_to_file(fobj)
                        fobj.seek(0)

                        # Image loading...
                        img_tensor = pil_to_tensor(Image.open(fobj))
                        imgs.append(self.transform(img_tensor))

                    # Stack...
                    assert len(imgs) == 2, "Something went awry with try/except in StateOK Dataset..."
                    imgs = torch.stack(imgs)

                    # Return --> based on `self.no_lang`
                    if not self.no_lang:
                        return imgs, lang, lang_mask
                    else:
                        return imgs

                except (NotFound, TransportError, UnidentifiedImageError, OSError) as e:
                    # At the minimum --> print the broken file (obnoxiously!)
                    print(f"=>> BROKEN FILE :: {current_frame}")
                    if not self.do_retry:
                        raise e
                    else:
                        continue

            # If we've exhausted our retries --> raise an informative ValueError
            raise ValueError(f"Failed to fix states `{self.elements[index]['states']}` w/ {self.n_retries} retries...")

    def __len__(self) -> int:
        return len(self.elements)


class GenStateOKDataset(PretrainDataset):
    def __init__(
        self,
        epoch: int,
        index_path: Path,
        img_transform: Compose,
        gen_ratio: float,
        stream: bool = False,
        prefix: Optional[Path] = None,
        is_val: bool = False,
        do_retry: bool = True,
        n_retries: int = 3,
    ) -> None:
        super().__init__()
        self.index_path, self.stream, self.is_val, self.val_loaded = index_path, stream, is_val, False
        self.epoch, self.transform, self.elements, self.prefix = epoch, img_transform, None, prefix
        self.gen_ratio, self.gen_indices = gen_ratio, set()
        self.r = N_CORES * get_rank()
        self.do_retry, self.n_retries = do_retry, n_retries

        # Load Language Index & Retrieve Epoch 0 Batches
        language_path = "val-language-index.json" if self.is_val else "train-language-index.json"
        if not self.stream:
            with open(self.index_path / language_path, "r") as f:
                self.language_index = json.load(f)
        else:
            blob = BUCKETS[self.r].blob(str(self.prefix / "index" / language_path))
            self.language_index = json.loads(blob.download_as_string())

        # === Retrieve Epoch Batches (only call `set_epoch` inside __init__() ===
        self.set_epoch(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        # Not Streaming --> Read from local disk...
        if not self.stream:
            if self.is_val and not self.val_loaded:
                with open(self.index_path / "state+ok" / "validation-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Assemble the set of dropout indices for the given epoch...
                n_gen = int(self.gen_ratio * len(self.elements))
                self.gen_indices = set(np.random.choice(len(self.elements), n_gen, replace=False))

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                with open(self.index_path / "state+ok" / f"train-epoch={epoch}-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Assemble the set of dropout indices for the given epoch...
                n_gen = int(self.gen_ratio * len(self.elements))
                self.gen_indices = set(np.random.choice(len(self.elements), n_gen, replace=False))

        # Streaming --> Beam directly from Bucket
        else:
            if self.is_val and not self.val_loaded:
                blob = BUCKETS[self.r].blob(str(self.prefix / "index" / "state+ok" / "validation-batches.json"))
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["states"]` currently maps to disk path.. remove all but `parent/child.jpg`
                for element in self.elements:
                    element["states"] = ["/".join(x.split("/")[-2:]) for x in element["states"]]

                # Assemble the set of dropout indices for the given epoch...
                n_gen = int(self.gen_ratio * len(self.elements))
                self.gen_indices = set(np.random.choice(len(self.elements), n_gen, replace=False))

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                blob = BUCKETS[self.r].blob(
                    str(self.prefix / "index" / "state+ok" / f"train-epoch={epoch}-batches.json")
                )
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["states"]` currently maps to disk path.. remove all but `parent/child.jpg`
                for element in self.elements:
                    element["states"] = ["/".join(x.split("/")[-2:]) for x in element["states"]]

                # Assemble the set of dropout indices for the given epoch...
                n_gen = int(self.gen_ratio * len(self.elements))
                self.gen_indices = set(np.random.choice(len(self.elements), n_gen, replace=False))

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Return both states/frames, language to condition on, language to generate, decomposed as input_ids/mask."""
        vid = self.elements[index]["vid"]

        # Fetch language to condition on / generate...
        lang_condition = torch.tensor(self.language_index[vid]["input_ids"], dtype=torch.int64)
        lang_condition_mask = torch.tensor(self.language_index[vid]["attention_mask"], dtype=torch.int64)
        lang_gen = torch.tensor(self.language_index[vid]["input_ids"], dtype=torch.int64)
        lang_gen_mask = torch.tensor(self.language_index[vid]["attention_mask"], dtype=torch.int64)

        # Generate/Condition Check --> (Naive Zeroing leads to NaN --> just want the "CLS" token)
        if index in self.gen_indices:
            # If generating, just condition on the <CLS> token (always the initial...), but generate everything!
            lang_condition[1:] *= 0
            lang_condition_mask[1:] *= 0
            lang_gen_weight = 1

        else:
            # If conditioning, generate the <CLS> token (dummy so things don't crash), but set weight to 0
            lang_gen[1:] *= 0
            lang_gen_mask[1:] *= 0
            lang_gen_weight = 0

        # Retrieve Frames
        if not self.stream:
            imgs = self.elements[index]["states"]
            imgs = torch.stack([self.transform(read_image(s)) for s in imgs])

            # Return...
            return imgs, lang_condition, lang_condition_mask, lang_gen, lang_gen_mask, lang_gen_weight

        else:
            # Multiplex w/ num_worker idx...
            worker_info = get_worker_info()
            r = (self.r + worker_info.id) if worker_info is not None else self.r

            # Streaming + Retry Logic (in case of a bad connection -- retry same files!)
            frame_paths, current_frame = list(self.elements[index]["states"]), None
            for _i in range(self.n_retries):
                try:
                    # Stream JPEG contents into BytesIO (seek back to 0), then PIL Image.open()
                    imgs = []
                    for _current_idx, current_frame in enumerate(frame_paths):
                        if self.is_val:
                            blob, fobj = BUCKETS[r].blob(str(self.prefix / "val" / current_frame)), BytesIO()
                        else:
                            blob, fobj = BUCKETS[r].blob(str(self.prefix / "train" / current_frame)), BytesIO()

                        # Error Handling --> we've already run several dry-run/verification trials, but about ~0.004% of
                        #                    the time, we'll hit some sort of TCP/Transport error; this might even go up
                        #                    with multiple runs happening at the same time.
                        #
                        #                    To address this, we're adopting the simplest possible "retry" strategy that
                        #                    immediately tries to re-download the same file (crashes if not possible).
                        #                    This ensures reproducibility, but puts some extra effort onto the user...

                        # File download...
                        blob.download_to_file(fobj)
                        fobj.seek(0)

                        # Image loading...
                        img_tensor = pil_to_tensor(Image.open(fobj))
                        imgs.append(self.transform(img_tensor))

                    # Stack...
                    assert len(imgs) == 2, "Something went awry with try/except in GenStateOK Dataset..."
                    imgs = torch.stack(imgs)

                    # Return...
                    return imgs, lang_condition, lang_condition_mask, lang_gen, lang_gen_mask, lang_gen_weight

                except (NotFound, TransportError, UnidentifiedImageError, OSError) as e:
                    # At the minimum --> print the broken file (obnoxiously!)
                    print(f"=>> BROKEN FILE :: {current_frame}")
                    if not self.do_retry:
                        raise e
                    else:
                        continue

            # If we've exhausted our retries --> raise an informative ValueError
            raise ValueError(f"Failed to fix states `{self.elements[index]['states']}` w/ {self.n_retries} retries...")

    def __len__(self) -> int:
        return len(self.elements)


class QuintetDataset(PretrainDataset):
    def __init__(
        self,
        epoch: int,
        index_path: Path,
        img_transform: Compose,
        lang_dropout: Optional[float] = None,
        stream: bool = False,
        prefix: Optional[Path] = None,
        is_val: bool = False,
    ) -> None:
        super().__init__()
        self.index_path, self.stream, self.is_val, self.val_loaded = index_path, stream, is_val, False
        self.epoch, self.transform, self.elements, self.prefix = epoch, img_transform, None, prefix
        self.lang_dropout = 0.0 if (lang_dropout is None or lang_dropout == 0) else lang_dropout
        self.dropout_indices = set()
        self.r = N_CORES * get_rank()

        # Load Language Index & Retrieve Epoch 0 Batches
        language_path = "val-language-index.json" if self.is_val else "train-language-index.json"
        if not self.stream:
            with open(self.index_path / language_path, "r") as f:
                self.language_index = json.load(f)
        else:
            blob = BUCKETS[self.r].blob(str(self.prefix / "index" / language_path))
            self.language_index = json.loads(blob.download_as_string())

        # === Retrieve Epoch Batches (only call `set_epoch` inside __init__() ===
        self.set_epoch(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        # Not Streaming --> Read from local disk...
        if not self.stream:
            if self.is_val and not self.val_loaded:
                with open(self.index_path / "quintet+language" / "validation-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                with open(self.index_path / "quintet+language" / f"train-epoch={epoch}-batches.json", "r") as f:
                    self.elements = json.load(f)

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

        # Streaming --> Beam directly from Bucket
        else:
            if self.is_val and not self.val_loaded:
                blob = BUCKETS[self.r].blob(str(self.prefix / "index" / "quintet+language" / "validation-batches.json"))
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["states"]` currently maps to disk path.. remove all but `parent/child.jpg`
                for element in self.elements:
                    element["states"] = ["/".join(x.split("/")[-2:]) for x in element["states"]]

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

                # Set `val_loaded` and move on...
                self.val_loaded = True

            elif not self.is_val:
                blob = BUCKETS[self.r].blob(
                    str(self.prefix / "index" / "quintet+language" / f"train-epoch={epoch}-batches.json")
                )
                self.elements = json.loads(blob.download_as_string())

                # `elements[i]["states"]` currently maps to disk path.. remove all but `parent/child.jpg`
                for element in self.elements:
                    element["states"] = ["/".join(x.split("/")[-2:]) for x in element["states"]]

                # Assemble the set of dropout indices for the given epoch...
                n_drop = int(self.lang_dropout * len(self.elements))
                self.dropout_indices = set(np.random.choice(len(self.elements), n_drop, replace=False))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all 5 states/frames, and language, decomposed as the input_ids and attention_mask."""
        vid = self.elements[index]["vid"]
        lang = torch.tensor(self.language_index[vid]["input_ids"], dtype=torch.int64)
        lang_mask = torch.tensor(self.language_index[vid]["attention_mask"], dtype=torch.int64)

        # Dropout Language Check --> (Naive Zeroing leads to NaN --> just want the "PAD" token)
        if index in self.dropout_indices:
            # Initial language token is *always* <CLS> = `101` --> last token always <SEP> = `102`
            lang[1:] *= 0
            lang_mask[1:] *= 0

        # Retrieve Frames
        if not self.stream:
            imgs = self.elements[index]["states"]
            imgs = torch.stack([self.transform(read_image(s)) for s in imgs])
        else:
            # Multiplex w/ num_worker idx...
            worker_info, imgs = get_worker_info(), []
            r = (self.r + worker_info.id) if worker_info is not None else self.r

            # Stream JPEG contents into BytesIO (seek back to 0), then PIL Image.open()
            for state in self.elements[index]["states"]:
                if self.is_val:
                    blob, fobj = BUCKETS[r].blob(str(self.prefix / "val" / state)), BytesIO()
                else:
                    blob, fobj = BUCKETS[r].blob(str(self.prefix / "train" / state)), BytesIO()

                # Download into FileObj & Rewind...
                blob.download_to_file(fobj)
                fobj.seek(0)

                # Add to imgs...
                imgs.append(self.transform(pil_to_tensor(Image.open(fobj))))

            # Stack...
            imgs = torch.stack(imgs)

        return imgs, lang, lang_mask

    def __len__(self) -> int:
        return len(self.elements)


def get_epoch_datasets(
    epoch: int,
    name: str,
    normalization: Tuple[Any, Any],
    model_arch: str,
    stream: bool,
    artifact_path: str,
    stream_prefix: str,
    data_modality: str,
    lang_dropout: Optional[float] = None,
    gen_ratio: Optional[float] = None,
) -> Tuple[PretrainDataset, PretrainDataset]:
    """Retrieve the custom Dataset classes for the train & val set for the given dataset & data modality."""
    index, img_transform = Path(artifact_path) / name / "index", get_online_transform(name, model_arch, normalization)
    prefix = Path(stream_prefix) / name if stream else stream_prefix

    # Switch on `data_modality`
    if data_modality == "state":
        train_ds = StateDataset(epoch, index, img_transform, stream, prefix)
        val_ds = StateDataset(epoch, index, img_transform, stream, prefix, is_val=True)

    elif data_modality == "state+language":
        train_ds = StateLanguageDataset(epoch, index, img_transform, lang_dropout, stream, prefix)
        val_ds = StateLanguageDataset(epoch, index, img_transform, lang_dropout, stream, prefix, is_val=True)

    elif data_modality == "state+ok":
        if gen_ratio is None:
            nl = model_arch == "v-dual"
            train_ds = StateOKDataset(epoch, index, img_transform, lang_dropout, stream, prefix, no_lang=nl)
            val_ds = StateOKDataset(epoch, index, img_transform, lang_dropout, stream, prefix, no_lang=nl, is_val=True)
        else:
            # Special Generative Language Dataset...
            train_ds = GenStateOKDataset(epoch, index, img_transform, gen_ratio, stream, prefix)
            val_ds = GenStateOKDataset(epoch, index, img_transform, gen_ratio, stream, prefix, is_val=True)

    elif data_modality == "quintet+language":
        train_ds = QuintetDataset(epoch, index, img_transform, lang_dropout, stream, prefix)
        val_ds = QuintetDataset(epoch, index, img_transform, lang_dropout, stream, prefix, is_val=True)

    else:
        raise NotImplementedError(f"Support for data modality `{data_modality}` not yet implemented!")

    return train_ds, val_ds
