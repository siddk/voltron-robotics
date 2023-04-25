"""
process.py

Utility functions for serializing datasets in multiple passes, using multiprocessing for efficient parallelization.
Exposes a three-phase sequence for preprocessing:
    - Phase I: Read in raw videos (and language), serialize *all extracted* frames to a subdirectory for easy retrieval.
    - Phase II: Given image paths and language, assemble language statistics & pre-tokenize for easy batching.
    - Phase III: Given a total number of "conceivable epochs", create data-controlled "epoch" sets for each model.

This script tries to be smart where it can, using multiprocessing.Pool in Phase I to speed up the serialization
process. It also tries to be somewhat safe & efficient, producing idempotent resumes.

Note :: This code represents the `v1` (initial release) preprocessing flow; this will eventually be deprecated!
"""
import json
import logging
import multiprocessing as mp
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Tuple

import torch
from rich.progress import track
from transformers import AutoTokenizer

from voltron.preprocessing.v1.transforms import get_pre_transform
from voltron.preprocessing.v1.utils import do_dry_run, precompute_epoch, process_video

# Grab Logger
overwatch = logging.getLogger(__file__)


def preprocess_videos(
    name: str,
    path: str,
    artifact_path: str = "data/processed",
    resolution: int = 224,
    n_val_videos: int = 1000,
    dry_run: bool = False,
) -> Tuple[Path, Path, Path, Path]:
    """Phase I of Preprocessing :: Uses Multiprocessing to Read Videos & Serialize Frames."""
    overwatch.info(f"Phase 1 Preprocessing :: Frame serializing videos for dataset `{name}`")

    if name == "sth-sth-v2":
        # Overview of Return Values:
        #   `t_registry` and `v_registry` =>> store mappings of "vid_id" -> {metadata}
        #   `t_dir` and `v_dir` =>> store "processed data" (extracted frames)
        t_dir, v_dir = Path(artifact_path) / name / "train", Path(artifact_path) / name / "val"
        t_registry, v_registry = t_dir / "registry.json", v_dir / "registry.json"

        # Short-Circuit / Caching Logic
        if t_registry.exists() and v_registry.exists():
            return t_registry, v_registry, t_dir, v_dir

        # Setup / Book-Keeping
        os.makedirs(t_dir, exist_ok=True)
        os.makedirs(v_dir, exist_ok=True)

        # Retrieve Image Transforms (pre-serialization, while running "offline" pass); we crop and scale once, so we're
        #   not overdoing it on disk storage...
        pre_transform = get_pre_transform(name, resolution=resolution)

        # Open & Extract Video ID & Language Metadata
        with open(Path(path) / "something-something-v2-train.json", "r") as f:
            annotations = json.load(f)
            train_ids, train_lang = [x["id"] for x in annotations], [x["label"] for x in annotations]

        with open(Path(path) / "something-something-v2-validation.json", "r") as f:
            annotations = json.load(f)[:n_val_videos]
            val_ids, val_lang = [x["id"] for x in annotations], [x["label"] for x in annotations]

        # Do Dry-Run --> Single-Threaded!
        if dry_run:
            do_dry_run(
                name,
                path,
                n_train_videos=1000,
                n_val_videos=100,
                train_ids=train_ids,
                val_ids=val_ids,
                pre_transform=pre_transform,
            )

        # Go Go Go =>> Iterate through all videos, dump all frames subject to the following structure:
        #   |-> data/processed/sth-sth-v2/
        #          |-> <split>/
        #                 |-> <video-id>/frames<0...k>.jpg
        # We'll track a single metadata file with the map of <video-id> : ("language", n_frames).
        #   > To speed up the serialization, we'll use a multiprocessing.Pool and max out CPU workers
        with mp.Pool(mp.cpu_count()) as pool:
            for k, save, vids, langs in [("train", t_dir, train_ids, train_lang), ("val", v_dir, val_ids, val_lang)]:
                overwatch.info(f"\tWriting `{k}` videos to disk...")

                # Multiprocess!
                process_fn, registration = partial(process_video, name, Path(path), save, pre_transform), {}
                for key, value in track(
                    pool.imap_unordered(process_fn, zip(vids, langs)),
                    description=f"\t[*] Processing {k}...",
                    total=len(vids),
                    transient=True,
                ):
                    if key is not None:
                        registration[key] = value

                # Write Registration to Disk
                with open(t_registry if k == "train" else v_registry, "w") as f:
                    json.dump(registration, f)

        # Return Paths...
        return t_registry, v_registry, t_dir, v_dir

    else:
        raise NotImplementedError(f"Preprocessing Pipeline for Dataset `{name}` not implemented!")


def preprocess_language(
    name: str, train_registry: Path, val_registry: Path, max_lang_len: int, language_model: str, hf_cache: str
) -> None:
    """Phase II of Preprocessing :: Iterate through Language & Normalize/Tokenize to Max Length."""
    overwatch.info(f"Phase 2 Preprocessing :: Normalizing & tokenizing language for dataset `{name}`")
    t_index, v_index = train_registry.parent / "index.pt", val_registry.parent / "index.pt"
    t_json, v_json = train_registry.parent / "index.json", val_registry.parent / "index.json"

    # Short-Circuit Logic
    if (t_index.exists() and v_index.exists()) or (t_json.exists() and v_json.exists()):
        return t_index, v_index

    # Grab Language, Retaining Metadata for Building Index Structures...
    with open(train_registry, "r") as f:
        train_metadata = json.load(f)
        train = [(vid, train_metadata[vid]["language"], train_metadata[vid]) for vid in train_metadata]

    with open(val_registry, "r") as f:
        val_metadata = json.load(f)
        val = [(vid, val_metadata[vid]["language"], val_metadata[vid]) for vid in val_metadata]

    # Assemble *all* language
    language = [x[1] for x in train + val]

    # Build AutoTokenizer (from `language_model` identifier)
    tokenizer = AutoTokenizer.from_pretrained(language_model, cache_dir=hf_cache)

    # If `max_lang_len` not specified, dump some statistics to compute...
    if max_lang_len == -1:
        # Naively tokenizes and pads to the "maximum length" of _all_ language... long tail is a problem!
        encoded_language = tokenizer(language, return_tensors="pt", padding=True)
        lengths = encoded_language["attention_mask"].sum(dim=1)

        # Compute a histogram of lengths
        hist = lengths.float().histc(bins=lengths.max()).int()
        overwatch.info(f"Histogram: {hist.numpy().tolist()}")
        raise NotImplementedError("Compute max length and update dataset configuration!")

    # Otherwise, we've already set the maximum length, so let's use it!
    else:
        overwatch.info(f"\tTokenizing all language in dataset to maximum length `{max_lang_len}`")
        encoded_language = tokenizer(
            language, return_tensors="pt", max_length=max_lang_len, truncation=True, padding="max_length"
        )
        input_ids, attention_mask = encoded_language["input_ids"], encoded_language["attention_mask"]
        train_input_ids, train_attention_mask = input_ids[: len(train)], attention_mask[: len(train)]
        val_input_ids, val_attention_mask = input_ids[len(train) :], attention_mask[len(train) :]

        # Assertion, just to sanity check
        assert len(val_input_ids) == len(val_attention_mask) == len(val), "Something went wrong tokenizing language..."

        # Compute `index.pt` contents
        overwatch.info("\tAssembling `train` and `val` index structures...")
        train_pt = {
            train[i][0]: {**train[i][2], **{"input_ids": train_input_ids[i], "attention_mask": train_attention_mask[i]}}
            for i in range(len(train))
        }
        val_pt = {
            val[i][0]: {**val[i][2], **{"input_ids": val_input_ids[i], "attention_mask": val_attention_mask[i]}}
            for i in range(len(val))
        }

        # Dump structures...
        overwatch.info(f"Saving index structures to `{t_index}` and `{v_index}` respectively...")
        torch.save(train_pt, t_index)
        torch.save(val_pt, v_index)


def jsonify_language(train_registry: Path, val_registry: Path) -> None:
    """Phase 2.5 (Aggregation) :: XLA is weird, won't load torch.Tensors in Dataset; JSONify instead."""
    overwatch.info("\tPhase 2 Aggregation :: JSONifying Language Index")
    t_index, v_index = train_registry.parent / "index.pt", val_registry.parent / "index.pt"
    t_json, v_json = train_registry.parent / "index.json", val_registry.parent / "index.json"
    train_json, val_json = {}, {}

    # Short-Circuit Logic
    if t_json.exists() and v_json.exists():
        return

    # Load Data, iterate through and "de-tensorize", while building up JSON symmetric structure...
    train_data, val_data = torch.load(t_index), torch.load(v_index)
    overwatch.info("JSONifying both Train and Validation")
    for vid in track(train_data, description="Train Language...", transient=True):
        train_json[vid] = {
            "language": train_data[vid]["language"],
            "n_frames": train_data[vid]["n_frames"],
            "input_ids": train_data[vid]["input_ids"].numpy().tolist(),
            "attention_mask": train_data[vid]["attention_mask"].numpy().tolist(),
        }
    for vid in track(val_data, description="Val Language...", transient=True):
        val_json[vid] = {
            "language": val_data[vid]["language"],
            "n_frames": val_data[vid]["n_frames"],
            "input_ids": val_data[vid]["input_ids"].numpy().tolist(),
            "attention_mask": val_data[vid]["attention_mask"].numpy().tolist(),
        }

    # Write Data to Disk
    overwatch.info("Writing JSON Indices")
    with open(t_json, "w") as f:
        json.dump(train_json, f)

    with open(v_json, "w") as f:
        json.dump(val_json, f)


def index(train_registry: Path, val_registry: Path, name: str, artifact_path: str = "data/processed") -> Path:
    """Phase 2.75 (Indexing) :: Pull out language.json & other `absolutely necessary` indices to separate directory."""
    overwatch.info("\tPhase 2 Indexing :: Indexing Language & Registry Files =>> Extracting to Separate Directory")

    # Create "index" directory...
    index_dir = Path(artifact_path) / name / "index"
    os.makedirs(index_dir, exist_ok=True)

    # Short-Circuit Logic
    if (index_dir / "train-language-index.json").exists() and (index_dir / "val-language-index.json").exists():
        return index_dir

    # Retrieve Language JSON indices (train & validation) & copy to new directory...
    t_json, v_json = train_registry.parent / "index.json", val_registry.parent / "index.json"
    shutil.copy(t_json, index_dir / "train-language-index.json")
    shutil.copy(v_json, index_dir / "val-language-index.json")

    return index_dir


def unify_batches(
    artifact_path: Path,
    name: str,
    train_registry: Path,
    val_registry: Path,
    train_dir: Path,
    val_dir: Path,
    index_dir: Path,
    batch_formats: Tuple[Tuple[str, Tuple[str, ...]], ...],
    max_epochs: int = 400,
    initial_final_alpha: float = 0.2,
) -> None:
    """Phase III of Preprocessing :: Assemble Batches for *all models* for *all epochs* in a consistent manner."""
    overwatch.info("Phase 3 Preprocessing :: Assembling Data-Equivalent Epochs for each Model Format")

    # Load Registry Files
    with open(train_registry, "r") as f:
        train_registrations = json.load(f)

    with open(val_registry, "r") as f:
        val_registrations = json.load(f)

    # Assert last element of `batch_formats` assumes all prior subsets...
    full_set_inputs = set(batch_formats[-1][1])
    for _, subset_inputs in batch_formats[:-1]:
        assert full_set_inputs.issuperset(set(subset_inputs)), "We have a problem with batch formats..."

    # Assemble Tracking Data
    b_keys, unique_states = {b[0] for b in batch_formats}, set()

    # Parse out all "state"-specific elements...
    state_elements = [s for s in full_set_inputs if "state_" in s]
    do_initial, do_final = "state_initial" in state_elements, "state_final" in state_elements
    n_int = len(state_elements) - 2 if ("state_initial" in state_elements and "state_final" in state_elements) else 0

    # Serialize Epochs to Disk
    overwatch.info("\tSerializing epochs to json file, pointing to image paths on disk via a dictionary...")
    for b in b_keys:
        os.makedirs(index_dir / b, exist_ok=True)

    # We only write the validation epoch once --> held constant across _all_ of training!
    overwatch.info("\tWriting Validation Epoch to Disk...")
    val_epoch_idx, _, uniq_s = precompute_epoch(
        index_dir,
        val_registrations,
        val_dir,
        batch_formats,
        do_initial,
        do_final,
        initial_final_alpha,
        n_int,
        0,
        is_validation=True,
    )

    # Update Trackers...
    if val_epoch_idx != -1:
        unique_states |= uniq_s

    # Compute length of epochs --> CPU Count should be no higher...
    epochs, n_frames_per_epoch = list(range(max_epochs)), -1

    # Load "existing" verification file (if possible)
    overwatch.info("\tLoading batch verification file (if possible)...")
    verified_batches = Path(artifact_path) / name / "verified-batches.json"
    if verified_batches.exists():
        with open(verified_batches, "r") as f:
            missing_epochs_per_format = json.load(f)

        # Set epochs list by taking union of missing epochs over formats...
        epochs = sorted(list(set().union(*missing_epochs_per_format.values())))

    # Dump the big objects into an mp.Manager() so that we can read efficiently from other workers...
    overwatch.info("\tPlacing the Train Registry into Shared Memory...")
    manager = mp.Manager()
    mg_registry = manager.dict(train_registrations)

    with mp.Pool(4) as pool:
        overwatch.info("\tWriting Train Batches per Epoch to Disk...")

        # Create partial function for multiprocessing pool...
        precompute_fn = partial(
            precompute_epoch,
            index_dir,
            mg_registry,
            train_dir,
            batch_formats,
            do_initial,
            do_final,
            initial_final_alpha,
            n_int,
        )
        for epoch_idx, n_frames, uniq_s in pool.imap_unordered(precompute_fn, epochs):
            if epoch_idx == -1:
                continue

            # Update Trackers
            unique_states |= uniq_s
            n_frames_per_epoch = n_frames

    # Statistics only make sense on initial computation... should unify with code above!
    overwatch.info(f"Train Uniqueness: {len(unique_states)} States & {len(mg_registry)} Utterances")
    overwatch.info(f"Final Statistics :: 1 Epoch has ~ {n_frames_per_epoch} Frames...")
    overwatch.info("Preprocessing Complete!")
