"""
utils.py

Preprocessing utilities, including functions for dry-runs and processing a single video (helpers for multiprocessing
calls down the lines).
"""
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import av
import cv2
import numpy as np
from hurry.filesize import alternative, size
from rich.progress import track
from tqdm import tqdm

from voltron.preprocessing.v1.transforms import ComposeMix

# Grab Logger
overwatch = logging.getLogger(__file__)
logging.getLogger("libav").setLevel(logging.ERROR)


# Videos are saved as `train_dir/{vid}/{vid}_idx={i}.jpg
def get_path(save_dir: Path, v: str, i: int) -> str:
    return str(save_dir / v / f"{v}_idx={i}.jpg")


def do_dry_run(
    name: str,
    path: str,
    n_train_videos: int,
    n_val_videos: int,
    train_ids: List[str],
    val_ids: List[str],
    pre_transform: ComposeMix,
    n_samples: int = 1000,
) -> None:
    """Iterates through a small subset of the total dataset, logs n_frames & average image size for estimation."""
    dry_run_metrics = {
        "n_frames": [],
        "jpg_sizes": [],
        "n_samples": n_samples,
        "time_per_example": [],
        "blank": str(Path(path) / "blank.jpg"),
    }
    if name == "sth-sth-v2":
        for k, n_iter, vids in [("train", n_train_videos, train_ids), ("val", n_val_videos, val_ids)]:
            for idx in track(range(n_iter), description=f"Reading {k.capitalize()} Videos =>> ", transient=True):
                vid = vids[idx]
                container = av.open(str(Path(path) / "videos" / f"{vid}.webm"))
                try:
                    imgs = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
                except (RuntimeError, ZeroDivisionError) as e:
                    overwatch.error(f"{type(e).__name__}: WebM reader cannot open `{vid}.webm` - continuing...")
                    continue

                # Close container
                container.close()

                # Apply `pre_transform`
                imgs = pre_transform(imgs)

                # Dry-Run Handling --> write a dummy JPEG to collect size statistics, dump, and move on...
                dry_run_metrics["n_frames"].append(len(imgs))
                while dry_run_metrics["n_samples"] > 0 and len(imgs) > 0:
                    img = imgs.pop(0)
                    cv2.imwrite(str(dry_run_metrics["blank"]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    dry_run_metrics["jpg_sizes"].append(os.path.getsize(dry_run_metrics["blank"]))
                    dry_run_metrics["n_samples"] -= 1

        # Compute nice totals for "dry-run" estimation
        total_clips = len(train_ids) + len(val_ids)

    else:
        raise NotImplementedError(f"Dry Run for Dataset `{name}` not yet implemented!")

    # Compute Aggregate Statistics and gently exit...
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
    sys.exit(0)


def process_video(
    name: str, path: Path, save: Path, pre_transform: ComposeMix, item: Tuple[str, str]
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Processes a single video file, dumps to series of image files, and returns the registry contents."""
    if name == "sth-sth-v2":
        # For sth-sth-v2, `item` corresponds to a single video clip, so just a tuple!
        vid, lang = item
        container, registration = av.open(str(Path(path) / "videos" / f"{vid}.webm")), {"language": lang, "n_frames": 0}
        try:
            imgs = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        except (RuntimeError, ZeroDivisionError) as e:
            overwatch.error(f"{type(e).__name__}: WebM reader cannot open `{vid}.webm` - skipping...")
            return None, None

        # Close container
        container.close()

        # Book-keeping
        os.makedirs(save / vid, exist_ok=True)
        registration["n_frames"] = len(imgs)

        # Early exit (writes are expensive)
        if len(glob.glob1(save / vid, "*.jpg")) == len(imgs):
            return vid, registration

        # Apply `pre_transform` --> write individual frames, register, and return
        imgs = pre_transform(imgs)
        for i in range(len(imgs)):
            cv2.imwrite(get_path(save, vid, i), cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))

        # Return title & registration
        return vid, registration

    else:
        raise NotImplementedError(f"Process Video for Dataset `{name}` not yet implemented!")


# ruff: noqa: C901
def precompute_epoch(
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

    # Short-Circuit
    if all([(index_dir / key / index_file).exists() for key, _ in batch_formats]):
        return -1, -1, None

    # Random seed is inherited from parent process... we want new randomness w/ each process
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Create Tracking Variables
    unique_states, batches = set(), {b: [] for b, _ in batch_formats}

    # Iterate through Registry...
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
        retrieved_states = [get_path(vid_dir, vid, x) for x in [initial_idx, *sampled_idxs] + [final_idx]]

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

    return epoch, len(batches["state"]), unique_states
