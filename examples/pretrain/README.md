# Pretraining Voltron Models

We provide scripts for pretraining Voltron models on various datasets. Below, we provide the full pipeline from
downloading the raw Something-Something-v2 Dataset from Qualcomm, running preprocessing, then running Distributed
Data Parallel (DDP) pretraining on 1+ GPUs via `torchrun`. Adding support for new datasets should follow this same
general flow.

---

## Dataset Preprocessing

We provide end-to-end instructions for downloading, preprocessing, and serializing various pretraining datasets (and
combinations thereof). Where possible, we provide links to batch/dataset index files.

**Note:** We make a key assumption that you have enough local disk space (e.g., on your server, attached NFS volume) to
store all *raw* and *preprocessed* data; this can range from 100s of GBs to 10s of TBs! We did not have access to such
storage in the original work, necessitating the *streaming* dataloaders defined in
`voltron/datasets/v1/stream_datasets.py`. Given your resources, you might consider adopting a similar approach; feel
free to post an issue with any questions!

We currently support pretraining on the following datasets:

- [Something-Something-v2](https://developer.qualcomm.com/software/ai-datasets/something-something)

Instructions for downloading/preprocessing each dataset can be found below!

---

### Something-Something-v2

Dataset Download: [Qualcomm AI Datasets](https://developer.qualcomm.com/software/ai-datasets/something-something)

#### Obtaining the Raw Dataset

Follow the instructions [at the above link](https://developer.qualcomm.com/software/ai-datasets/something-something) to
download the dataset. Qualcomm requires that you register for a
[Qualcomm OneID Account](https://myaccount.qualcomm.com/signup?target=https%3A%2F%2Fdeveloper.qualcomm.com)
to get access to the data. Approval might take some time.

After registering for an account, make sure to download all of the following files to a directory of your choosing
(we create a directory `data/raw/something-something-v2/downloaded/`). *You will need to manually download all 22 of
the following files from the Qualcomm site*:

1. Datasheet / Instructions (PDF – optional, but useful): `20bn-something-something_download_instructions_-_091622.pdf`
2. Labels (includes language annotations): `20bn-something-something_download-package-labels.zip`
3. Chunked Videos (should be 20 `.zip` archives):
   + `20bn-something-something-v2-00.zip`
   + ...
   + `20bn-something-something-v2-19.zip`

To extract all the given files (we extract to `data/raw/something-something-v2/`) - *execute the following from inside
the `downloaded/` subdirectory)*:

```bash
# Labels (annotations/language) --> creates `data/raw/something-something-v2/labels`
unzip 20bn-something-something-download-package-labels.zip -d ../

# Videos (following instructions in `20-bn-something-something_download_instructions_-_091622.pdf`)
unzip "20bn-something-something-v2-*.zip" -d ../videos
cd ../videos
cat 20bn-something-something-?? | tar -xvzf -
find . -maxdepth 1 -type f -delete
cd 20bn-something-something-v2/
find . -mindepth 1 -maxdepth 1 -exec mv -t .. -- {} +
cd ..
rm -r 20bn-something-something-v2
ls | wc   # Should have 220847 `.webm` files!
```

#### Dataset Information & Statistics

Something-Something-v2 consists of 220,847 `.webm` clips (168,913 in the `train` split) each with a height of exactly
240px, and variable width. The frames are encoded at a fixed 12 FPS.

There are an average of 45 frames per clip (approx ~7 KB per jpeg); ~7.6M frames total (~56 GB).

#### Video/Image Transformations --> from Video Clip to "frame" --> "tensor"

```python
import av
from PIL import Image, ImageOps

# Resolutions for "preprocessing" (serialize to disk) and "training"
PREPROCESS_RESOLUTION, TRAIN_RESOLUTION = 240, 224

# Define Preprocessing Transformation
def preprocess_transform(frames: List[Image.Image]) -> List[Image.Image]:
    # Assert width >= height and height >= PREPROCESS_RESOLUTION
    orig_w, orig_h = frames[0].size
    assert orig_w >= orig_h >= PREPROCESS_RESOLUTION

    # Compute scale factor --> just a function of height and PREPROCESS_RESOLUTION
    scale_factor = PREPROCESS_RESOLUTION / orig_h

    # Full Transformation --> scale (preserve aspect ratio, then get square)
    for idx in range(len(frames)):
        frames[idx] = ImageOps.scale(frames[idx], factor=scale_factor)
        left = (frames[idx].size[0] - PREPROCESS_RESOLUTION) // 2
        frames[idx] = frames[idx].crop((left, 0, left + PREPROCESS_RESOLUTION, PREPROCESS_RESOLUTION))

    return frames

def train_transform(img) -> torch.Tensor:
    # Assumes square, just resizes to TRAIN_RESOLUTION via `torchvision.transforms`
    ...

def extract_frames(webm_file: str) -> None:
    container = av.open(webm_file)
    assert int(container.streams.video[0].average_rate) == 12, "FPS for `sth-sth-v2` should be 12!"

    # Extract --> then serialize via `Image.save("frame_{idx}.jpg")`
    frames = preprocess_transform([f.to_image() for f in container.decode(video=0)])
    ...
```


#### Citation

If you are pretraining on this dataset, make sure to cite the original research; Something-Something-v2 is the product
of two papers:

```bibtex
@inproceedings{goyal2017sthsthv1,
  author = {Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and Joanna Materzynska and Susanne Westphal and Heuna Kim and Valentin Haenel and Ingo Fründ and Peter N. Yianilos and Moritz Mueller-Freitag and Florian Hoppe and Christian Thurau and Ingo Bax and Roland Memisevic},
  booktitle = {International Conference on Computer Vision (ICCV)},
  title = {The ``Something Something'' Video Database for Learning and Evaluating Visual Common Sense},
  year = {2017},
}
@article{mahidisoltani2018sthsthv2,
  author={Farzaneh Mahdisoltani and Guillaume Berger and Waseem Gharbieh and David J. Fleet and Roland Memisevic},
  journal = {arXiv preprint arXiv:1804.09235},
  title={On the Effectiveness of Task Granularity for Transfer Learning},
  year={2018}
}
```

---

## PyTorch Native Pretraining Pipeline

To pretrain a Voltron model (e.g., `v-cond`) on the processed data, make sure to read `examples/pretrain/preprocess.py`.
A sample launch command to run with the Something-Something-v2 dataset on a single node with 8 GPUs is as follows:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 examples/pretrain/pretrain.py
```

Make sure to check the following configuration files and either update them manually (adding your own dataclass,
overriding [DEFAULTS](https://github.com/siddk/voltron-robotics/blob/main/examples/pretrain/pretrain.py#L38)), or by
using Hydra semantics to override them at the command line (e.g., `... pretrain.py dataset.path="<PATH>" ...`):

- [Accelerator Config](../../voltron/conf/accelerators.py): Depending on hardware, might need to tune `num_workers`
- [Dataset Config](../../voltron/conf/datasets.py): Make sure to override `path` and `artifact_path`
- [Tracking Config](../../voltron/conf/tracking.py): Disable Weights & Biases / change default entity/name
