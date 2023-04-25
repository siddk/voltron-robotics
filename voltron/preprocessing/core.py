"""
utils.py

Preprocessing utilities, including dry-run and single-video (single-example) processing. This file effectively defines
the "atomic" logic (take one video --> extract all frames, etc.), while the `process.py` functions invoke each unit
in a multiprocessing pool.
"""
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import av
import h5py
import numpy as np
import pandas as pd
from hurry.filesize import alternative, size
from PIL import Image
from rich.progress import track
from tqdm import tqdm

# Grab Logger
overwatch = logging.getLogger(__file__)
logging.getLogger("libav").setLevel(logging.ERROR)


# === General Utilities ===


# Videos are saved as `train_dir/{vid}/{vid}_idx={i}.jpg || if `relpath` then *relative path* `{split}/{vid}/...
def get_path(save_dir: Path, v: str, i: int, relpath: bool = False) -> str:
    return str((save_dir if not relpath else Path(save_dir.name)) / v / f"{v}_idx={i}.jpg")


# === Dry-Run Functionality ===


def do_dry_run(
    name: str,
    path: str,
    train_ids: List[str],
    val_ids: List[str],
    preprocess_transform: Callable[[List[Image.Image]], List[Image.Image]],
    n_train_videos: int = 1000,
    n_val_videos: int = 100,
    n_samples: int = 1000,
) -> None:
    """Iterates through a small subset of the total dataset, logs n_frames & average image size for estimation."""
    overwatch.info(f"Performing Dry-Run with {n_train_videos} Train Videos and {n_val_videos} Validation Videos")
    dry_run_metrics = {
        "n_frames": [],
        "jpg_sizes": [],
        "n_samples": n_samples,
        "time_per_example": [],
        "blank": str(Path(path) / "blank.jpg"),
    }

    # Switch on dataset (`name`)
    if name == "sth-sth-v2":
        for k, n_iter, vids in [("train", n_train_videos, train_ids), ("val", n_val_videos, val_ids)]:
            for idx in track(range(n_iter), description=f"Reading {k.capitalize()} Videos =>> ", transient=True):
                container = av.open(str(Path(path) / "videos" / f"{vids[idx]}.webm"))
                assert int(container.streams.video[0].average_rate) == 12, "FPS for `sth-sth-v2` should be 12!"
                try:
                    imgs = [f.to_image() for f in container.decode(video=0)]
                except (RuntimeError, ZeroDivisionError) as e:
                    overwatch.error(f"{type(e).__name__}: WebM reader cannot open `{vids[idx]}.webm` - continuing...")
                    continue
                container.close()

                # Apply `preprocess_transform`
                imgs = preprocess_transform(imgs)

                # Dry-Run Handling --> write a dummy JPEG to collect size statistics, dump, and move on...
                dry_run_metrics["n_frames"].append(len(imgs))
                while dry_run_metrics["n_samples"] > 0 and len(imgs) > 0:
                    img = imgs.pop(0)
                    img.save(str(dry_run_metrics["blank"]))
                    dry_run_metrics["jpg_sizes"].append(os.path.getsize(dry_run_metrics["blank"]))
                    dry_run_metrics["n_samples"] -= 1

        # Compute nice totals for "dry-run" estimate...
        total_clips = len(train_ids) + len(val_ids)

    else:
        raise ValueError(f"Dry Run for Dataset `{name}` not implemented!")

    # Compute aggregate statistics and gently exit...
    avg_size, avg_frames = np.mean(dry_run_metrics["jpg_sizes"]), int(np.mean(dry_run_metrics["n_frames"]))
    overwatch.info("Dry-Run Statistics =>>")
    overwatch.info(f"\t> A video has on average `{avg_frames}` frames at {size(avg_size, system=alternative)}")
    overwatch.info(f"\t> So - 1 video ~ {size(avg_frames * avg_size, system=alternative)}")
    overwatch.info(
        f"\t> With the full dataset of {total_clips} Train + Val videos ~"
        f" {size(total_clips * avg_frames * avg_size, system=alternative)}"
    )
    overwatch.info("Dry-Run complete, do what you will... exiting ✌️")

    # Remove dummy file...
    os.remove(dry_run_metrics["blank"])
    exit(0)


# === Atomic "Processing" Steps ===


def process_clip(
    name: str,
    path: Path,
    save: Path,
    preprocess_transform: Callable[[List[Image.Image]], List[Image.Image]],
    item: Tuple[str, str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Processes a single video clip and extracts/serializes all frames (as jpeg), returning the registry contents."""
    if name == "sth-sth-v2":
        vid, lang = item
        container, registration = av.open(str(Path(path) / "videos" / f"{vid}.webm")), {"language": lang, "n_frames": 0}
        assert int(container.streams.video[0].average_rate) == 12, "FPS for `sth-sth-v2` should be 12!"
        try:
            imgs = [f.to_image() for f in container.decode(video=0)]
        except (RuntimeError, ZeroDivisionError) as e:
            overwatch.error(f"{type(e).__name__}: WebM reader cannot open `{vid}.webm` - continuing...")
            return None, None
        container.close()

        # Book-Keeping
        os.makedirs(save / vid, exist_ok=True)
        registration["n_frames"] = len(imgs)

        # Short Circuit --> Writes are Expensive!
        if len(glob.glob1(save / vid, "*.jpg")) == len(imgs):
            return vid, registration

        # Apply `preprocess_transform` --> write individual frames, register, and move on!
        imgs = preprocess_transform(imgs)
        for idx in range(len(imgs)):
            imgs[idx].save(get_path(save, vid, idx))

        # Return title & registration
        return vid, registration

    else:
        raise ValueError(f"Clip Processing for Dataset `{name}` is not implemented!")


# ruff: noqa: C901
def serialize_epoch(
    index_dir: Path,
    registry: Dict[str, Any],
    vid_dir: Path,
    batch_formats: Tuple[Tuple[str, Tuple[str, ...]], ...],
    do_initial: bool,
    do_final: bool,
    initial_final_alpha: float,
    n_int: int,
    epoch: int,
    is_validation: bool = False,
) -> Tuple[int, int, Optional[Set[str]]]:
    index_file = "validation-batches.json" if is_validation else f"train-epoch={epoch}-batches.json"
    index_hdf5 = "validation-batches.hdf5" if is_validation else f"train-epoch={epoch}-batches.hdf5"

    # Short-Circuit
    if all([(index_dir / key / index_file).exists() for key, _ in batch_formats]):
        return -1, -1, None

    # Random seed is inherited from parent process... we want new randomness w/ each process
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Create Tracking Variables
    unique_states, batches = set(), {b: [] for b, _ in batch_formats}

    # Iterate through Registry --> Note we're using `tqdm` instead of `track` here because of `position` feature!
    for vid in tqdm(registry.keys(), desc=f"Epoch {epoch}", total=len(registry), position=epoch):
        # The initial/final states are sampled from the first [0, \alpha) and final 1-\alpha, 1] percent of the video
        n_frames = registry[vid]["n_frames"]
        initial_idx, final_idx = 0, n_frames - 1
        if do_initial:
            initial_idx = np.random.randint(0, np.around(n_frames * initial_final_alpha))

        if do_final:
            final_idx = np.random.randint(np.around(n_frames * (1 - initial_final_alpha)), n_frames)

        # Assertion --> initial_idx < final_idx - len(state_elements)
        assert initial_idx < final_idx - n_int, "Initial & Final are too close... no way to sample!"

        # Assume remaining elements are just random "interior" states --> sort to get ordering!
        sampled_idxs = np.random.choice(np.arange(initial_idx + 1, final_idx), size=n_int, replace=False)
        sampled_idxs = sorted(list(sampled_idxs))

        # Compile full-set "batch"
        retrieved_states = [get_path(vid_dir, vid, x, relpath=True) for x in [initial_idx, *sampled_idxs] + [final_idx]]

        # Add batch to index for specific batch_format key...
        batches[batch_formats[-1][0]].append({"vid": vid, "states": retrieved_states, "n_frames": n_frames})
        unique_states.update(retrieved_states)

        # Add all other batch formats to indices...
        for key, elements in batch_formats[:-1]:
            n_states = len([x for x in elements if "state_" in x])
            assert (n_states <= 2) or (
                n_states == len(retrieved_states)
            ), f"Strange value of n_states={n_states} > 2 and not equal to total possible of {len(retrieved_states)}"

            # States are all independent -- each of the retrieved states is its own example...
            if n_states == 1:
                for idx in range(len(retrieved_states)):
                    batches[key].append({"vid": vid, "state": retrieved_states[idx], "n_frames": n_frames})

            # OK-Context is the only "valid" context for n_states == 2
            elif n_states == 2:
                assert elements == ["state_initial", "state_i", "language"], "n_states = 2 but not 0K context?"

                # Append 0th state to each of the remaining sampled contexts (usually 2 or 4)... each pair is an example
                for idx in range(1, len(retrieved_states)):
                    batches[key].append(
                        {"vid": vid, "states": [retrieved_states[0], retrieved_states[idx]], "n_frames": n_frames}
                    )

            # We're treating the entire sequence of retrieved states as a single example (for TCN/R3M/Temporal Models)
            else:
                batches[key].append({"vid": vid, "states": retrieved_states, "n_frames": n_frames})

    # Write JSON Index directly to disk...
    for key in batches:
        with open(index_dir / key / index_file, "w") as f:
            json.dump(batches[key], f)

    # Write HDF5 Index directly to disk...
    for key, elements in batch_formats[:-1]:
        n_states = len([x for x in elements if "state_" in x])

        # Create HDF5 File
        df = pd.DataFrame(batches[key])
        h5 = h5py.File(index_dir / key / index_hdf5, "w")
        for k in ["vid", "n_frames"]:
            h5.create_dataset(k, data=df[k].values)

        # Handle "state(s)" --> (image path strings) --> add leading dimension (`n_states`)
        if n_states == 1:
            dfs = df["state"].apply(pd.Series)
            h5.create_dataset("states", data=dfs.values)

        else:
            dfs = df["states"].apply(pd.Series)
            h5.create_dataset("states", data=dfs.values)

        # Close HDF5 File
        h5.close()

    return epoch, len(batches["state"]), unique_states
