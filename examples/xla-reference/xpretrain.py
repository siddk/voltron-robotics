"""
xpretrain.py

(The `x` prefix indicates this is a script geared for XLA/TPU backends *only*)!

Reference script for PyTorch XLA (TPU-based) pretraining on the Something-Something-v2 dataset; this is
mostly for completeness =>> the hope is that the regular `pretrain.py` script is more general and maintained.

Focuses on multi-TPU (XLA) training --> but also supports single-core TPU training, as the default distributed mp.spawn
behavior just collapses into a single thread! Loads and preprocesses dataset, instantiates a model, and runs training.

Run with `python examples/xla-reference/xpretrain.py` (will use the configuration specified by `DEFAULTS` below).
"""
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import jsonlines
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as parallel
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from voltron.conf import AcceleratorConfig, DatasetConfig, ModelConfig, TrackingConfig
from voltron.datasets.v1.stream_datasets import get_epoch_datasets
from voltron.models import VMVP, VR3M, VRN3M, VCond, VDual, VGen
from voltron.overwatch import OverwatchRich
from voltron.util.v1.checkpointing import XLACheckpointSaver
from voltron.util.v1.distributed import ResumeableDistributedSampler
from voltron.util.v1.random import set_global_seed
from voltron.util.v1.xla_logger import (
    log_epoch_end_update,
    log_vcond_train_update,
    log_vdual_train_update,
    log_vgen_train_update,
    log_vmvp_train_update,
    log_vr3m_train_update,
    log_vrn3m_train_update,
)

# Set Defaults (Hydra w/ Structured Configs)
DEFAULTS = [
    "_self_",
    {"model": "v-cond"},
    {"dataset": "sth-sth-v2"},
    {"accelerator": "tpu-v3-8"},
    {"tracking": "voltron-tracking"},
    {"override hydra/job_logging": "overwatch_rich"},
]


@dataclass
class PretrainConfig:
    # fmt: off
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: Dict[str, Any] = field(default_factory=lambda: {
        "run": {"dir": "./runs/train/${model.identifier}+dataset-${dataset.name}"}
    })

    # Command Line Arguments
    run_id: Optional[str] = None                                        # Run ID for Logging
    seed: int = 21                                                      # Random Seed (for reproducibility)

    # Resume / Debug Behavior
    resume: bool = True                                                 # Whether to resume an existing run...
    resume_epoch: Optional[int] = None                                  # Epoch to resume (if auto-resuming)...
    checkpoint_path: Optional[str] = None                               # Path to the specific checkpoint to load!
    wandb_resume_id: Optional[str] = None                               # W&B Run ID for `resume` behavior...

    # Composable / Structured Arguments
    model: ModelConfig = MISSING                                        # Model architecture for pretraining
    dataset: DatasetConfig = MISSING                                    # List of datasets for pretraining
    accelerator: AcceleratorConfig = MISSING                            # Accelerator configuration
    tracking: TrackingConfig = MISSING                                  # Run/experiment tracking configuration
    # fmt: on


# Hydra Setup :: Retrieve ConfigStore (Singleton) & Register Components
cs = ConfigStore.instance()
cs.store(group="hydra/job_logging", name="overwatch_rich", node=OverwatchRich)  # Annoying - configure logger for Hydra
cs.store(name="config", node=PretrainConfig)


# ruff: noqa: C901
def xpretrain(cfg: PretrainConfig) -> None:
    # Identify if `is_rank_zero` --> We only log from the rank zero process!
    is_rank_zero = xm.is_master_ordinal(local=False)
    xm.master_print("Voltron Training :: Assembling the Legendary Defender...")

    # Create Unique Run Name -- if `resume = True` we assume same "run_id"
    run_id = cfg.run_id
    if run_id is None:
        run_id = run_dir = f"{cfg.model.identifier}+{cfg.dataset.name}-x{cfg.seed}"
        cfg.run_id = run_id
    else:
        cfg.run_id = run_dir = run_id

    if is_rank_zero:
        os.makedirs(run_dir, exist_ok=True)

    xm.master_print(
        '\t=>> "If you get too worried about what could go wrong, you might miss a chance to do something great."'
    )

    # Set Randomness, get DataLoader worker initialization function (to ensure any random augmentations!)
    worker_init_fn = set_global_seed(cfg.seed)

    # Model Initialization Logic
    xm.master_print("Initializing Model and Placing on Different Devices...")
    if cfg.model.arch == "v-mvp":
        xm.master_print(f"Initializing MVP variant `{cfg.model.identifier}`")
        model = VMVP(
            resolution=cfg.dataset.resolution,
            patch_size=cfg.model.patch_size,
            encoder_depth=cfg.model.encoder_depth,
            encoder_embed_dim=cfg.model.encoder_embed_dim,
            encoder_n_heads=cfg.model.encoder_n_heads,
            decoder_depth=cfg.model.decoder_depth,
            decoder_embed_dim=cfg.model.decoder_embed_dim,
            decoder_n_heads=cfg.model.decoder_n_heads,
            optimizer=cfg.model.optimizer,
            schedule=cfg.model.schedule,
            base_lr=cfg.model.base_lr,
            min_lr=cfg.model.min_lr,
            effective_bsz=cfg.model.effective_bsz,
            betas=cfg.model.betas,
            weight_decay=cfg.model.weight_decay,
            warmup_epochs=cfg.dataset.warmup_epochs,
            max_epochs=cfg.dataset.max_epochs,
            mlp_ratio=cfg.model.mlp_ratio,
            norm_pixel_loss=cfg.model.norm_pixel_loss,
        )

    elif cfg.model.arch == "v-r3m":
        xm.master_print(f"Initializing R3M (ViT) Variant `{cfg.model.identifier}`")
        model = VR3M(
            resolution=cfg.dataset.resolution,
            patch_size=cfg.model.patch_size,
            depth=cfg.model.depth,
            embed_dim=cfg.model.embed_dim,
            n_heads=cfg.model.n_heads,
            language_model=cfg.model.language_model,
            hf_cache=cfg.model.hf_cache,
            language_dim=cfg.model.language_dim,
            reward_dim=cfg.model.reward_dim,
            n_negatives=cfg.model.n_negatives,
            lang_reward_weight=cfg.model.lang_reward_weight,
            tcn_weight=cfg.model.tcn_weight,
            l1_weight=cfg.model.l1_weight,
            l2_weight=cfg.model.l2_weight,
            optimizer=cfg.model.optimizer,
            schedule=cfg.model.schedule,
            lr=cfg.model.lr,
            min_lr=cfg.model.min_lr,
            warmup_epochs=cfg.dataset.warmup_epochs,
            max_epochs=cfg.dataset.max_epochs,
            mlp_ratio=cfg.model.mlp_ratio,
        )

    elif cfg.model.arch == "v-rn3m":
        xm.master_print(f"Intializing R3M (ResNet) Variant `{cfg.model.identifier}`")
        model = VRN3M(
            resolution=cfg.dataset.resolution,
            fc_dim=cfg.model.fc_dim,
            language_model=cfg.model.language_model,
            hf_cache=cfg.model.hf_cache,
            language_dim=cfg.model.language_dim,
            reward_dim=cfg.model.reward_dim,
            n_negatives=cfg.model.n_negatives,
            lang_reward_weight=cfg.model.lang_reward_weight,
            tcn_weight=cfg.model.tcn_weight,
            l1_weight=cfg.model.l1_weight,
            l2_weight=cfg.model.l2_weight,
            optimizer=cfg.model.optimizer,
            lr=cfg.model.lr,
        )

    elif cfg.model.arch == "v-cond":
        xm.master_print(f"Initializing Voltron V-Cond variant `{cfg.model.identifier}`")
        model = VCond(
            resolution=cfg.dataset.resolution,
            patch_size=cfg.model.patch_size,
            encoder_depth=cfg.model.encoder_depth,
            encoder_embed_dim=cfg.model.encoder_embed_dim,
            encoder_n_heads=cfg.model.encoder_n_heads,
            decoder_depth=cfg.model.decoder_depth,
            decoder_embed_dim=cfg.model.decoder_embed_dim,
            decoder_n_heads=cfg.model.decoder_n_heads,
            language_model=cfg.model.language_model,
            hf_cache=cfg.model.hf_cache,
            language_dim=cfg.model.language_dim,
            optimizer=cfg.model.optimizer,
            schedule=cfg.model.schedule,
            base_lr=cfg.model.base_lr,
            min_lr=cfg.model.min_lr,
            effective_bsz=cfg.model.effective_bsz,
            betas=cfg.model.betas,
            weight_decay=cfg.model.weight_decay,
            warmup_epochs=cfg.dataset.warmup_epochs,
            max_epochs=cfg.dataset.max_epochs,
            mlp_ratio=cfg.model.mlp_ratio,
            norm_pixel_loss=cfg.model.norm_pixel_loss,
        )

    elif cfg.model.arch == "v-dual":
        xm.master_print(f"Initializing Voltron V-Dual variant `{cfg.model.identifier}`")
        model = VDual(
            resolution=cfg.dataset.resolution,
            patch_size=cfg.model.patch_size,
            encoder_depth=cfg.model.encoder_depth,
            encoder_embed_dim=cfg.model.encoder_embed_dim,
            encoder_n_heads=cfg.model.encoder_n_heads,
            decoder_depth=cfg.model.decoder_depth,
            decoder_embed_dim=cfg.model.decoder_embed_dim,
            decoder_n_heads=cfg.model.decoder_n_heads,
            language_model=cfg.model.language_model,
            hf_cache=cfg.model.hf_cache,
            language_dim=cfg.model.language_dim,
            optimizer=cfg.model.optimizer,
            schedule=cfg.model.schedule,
            base_lr=cfg.model.base_lr,
            min_lr=cfg.model.min_lr,
            effective_bsz=cfg.model.effective_bsz,
            betas=cfg.model.betas,
            weight_decay=cfg.model.weight_decay,
            warmup_epochs=cfg.dataset.warmup_epochs,
            max_epochs=cfg.dataset.max_epochs,
            mlp_ratio=cfg.model.mlp_ratio,
            norm_pixel_loss=cfg.model.norm_pixel_loss,
        )

    elif cfg.model.arch == "v-gen":
        xm.master_print(f"Initializing Voltron V-Gen variant `{cfg.model.identifier}`")
        model = VGen(
            resolution=cfg.dataset.resolution,
            patch_size=cfg.model.patch_size,
            encoder_depth=cfg.model.encoder_depth,
            encoder_embed_dim=cfg.model.encoder_embed_dim,
            encoder_n_heads=cfg.model.encoder_n_heads,
            decoder_depth=cfg.model.decoder_depth,
            decoder_embed_dim=cfg.model.decoder_embed_dim,
            decoder_n_heads=cfg.model.decoder_n_heads,
            language_model=cfg.model.language_model,
            hf_cache=cfg.model.hf_cache,
            language_dim=cfg.model.language_dim,
            max_lang_len=cfg.dataset.max_lang_len,
            vocab_size=cfg.model.vocab_size,
            mae_weight=cfg.model.mae_weight,
            lm_weight=cfg.model.lm_weight,
            optimizer=cfg.model.optimizer,
            schedule=cfg.model.schedule,
            base_lr=cfg.model.base_lr,
            min_lr=cfg.model.min_lr,
            effective_bsz=cfg.model.effective_bsz,
            betas=cfg.model.betas,
            weight_decay=cfg.model.weight_decay,
            warmup_epochs=cfg.dataset.warmup_epochs,
            max_epochs=cfg.dataset.max_epochs,
            mlp_ratio=cfg.model.mlp_ratio,
            norm_pixel_loss=cfg.model.norm_pixel_loss,
        )

    else:
        raise NotImplementedError(f"Model Architecture `{cfg.model.arch}` is not supported!")

    # We use gradient accumulation to honor the effective batch size specified...
    assert cfg.model.effective_bsz % cfg.model.device_bsz == 0, "Device bsz must evenly divide effective bsz!"
    accumulate_grad_batches = cfg.model.effective_bsz // cfg.model.device_bsz // xm.xrt_world_size()
    xm.master_print(
        f"Running `{cfg.model.identifier}` model w/ Effective Batch Size of `{cfg.model.effective_bsz}`, "
        f"Per-Device Batch Size of `{cfg.model.device_bsz}`, "
        f"Distributed World Size of `{xm.xrt_world_size()}` and `{accumulate_grad_batches}` Accumulation Steps"
    )

    # If Resuming =>> Load Model from Checkpoint
    start_checkpoint, start_epoch, start_step = None, 0, 0
    if cfg.resume:
        # **IMPORTANT**: We're making a few assumptions on resuming that should eventually become explicit checks:
        #   - `accumulate_grad_batches` is exactly the same when resuming; this means:
        #       + `cfg.model.effective_bsz`, `cfg.model.device_bsz`, & `cfg.accelerator.num_accelerators` are the same!
        #   - The Weights & Biases directory `run_dir/wandb` only contains a *single run*
        #   - The `param_groups` in `optimizer.state_dict()` are exactly the same across resumes!
        #       + This means that (and generally should be true for resuming altogether) the architecture is the same!
        #   - The `cfg.seed` should be the same (again, should generally be true...)
        if cfg.checkpoint_path is None:
            xm.master_print("Resuming :: Attempting to Automatically Load Checkpoint -- Searching!")
            checkpoint_path = Path(run_dir) / "checkpoints"
            if checkpoint_path.exists() and any(checkpoint_path.iterdir()):
                # Parse out the latest "complete" epoch checkpoint, as well as any "local step" checkpoints...
                checkpoints = list(checkpoint_path.iterdir())
                complete_checkpoint, complete_epoch = max(
                    [
                        (c, int(re.search("epoch=(.+?)-train", c.name).group(1)))
                        for c in checkpoints
                        if "local-epoch=" not in str(c)
                    ],
                    key=lambda x: x[1],
                )

                # Case 1 :: We have "local step" checkpoints --> will always override any "full epoch" checkpoints...
                local = [
                    (
                        c,
                        int(re.search("local-epoch=(.+?)-step", c.name).group(1)),
                        int(re.search("step=(.+?)[.-]", c.name).group(1)),
                    )
                    for c in checkpoints
                    if "local-epoch=" in str(c)
                ]
                if len(local) > 0:
                    # Parse out (epoch, "highest" step) + assert no great "full epoch" checkpoint exists!
                    start_checkpoint, start_epoch, start_step = max(local, key=lambda x: x[1:])
                    assert start_epoch == complete_epoch, "Epoch mismatch in `resume` from local_step!"

                # Case 2 :: Otherwise, we're just going to start with the last "complete" epoch...
                else:
                    start_checkpoint, start_epoch = complete_checkpoint, complete_epoch

            else:
                xm.master_print("No Checkpoints Found -- Starting Run from Scratch!")

        else:
            xm.master_print(f"Resuming :: Loading from Checkpoint `{cfg.checkpoint_path}`...")
            start_checkpoint = cfg.checkpoint_path

        # Actually Load the Checkpoint State!
        if start_checkpoint is not None:
            xm.master_print(f"Resuming :: Loading Model & Optimizer State Dictionaries from `{start_checkpoint}`")
            checkpoint = torch.load(str(start_checkpoint))
            model_state_dict, optimizer_state_dict = checkpoint
            model.load_state_dict(model_state_dict)

    # Logging / W&B Handling
    if is_rank_zero:
        xm.master_print("Initializing Weights & Biases + JSONL + Checkpoint Saver on Rank Zero ONLY...")
        tags = None
        if cfg.tracking.tags is None:
            tags = [cfg.model.identifier, cfg.dataset.name, "pretraining"]

        # W&B Initialize & Log all Hyperparameters (Only on ordinal 0)
        wandb_resume_id = None
        if cfg.resume and cfg.wandb_resume_id is None:
            xm.master_print("Resuming :: Attempting to Automatically Load W&B Resume ID -- Searching!")
            wandb_path = Path("wandb")
            if wandb_path.exists() and any((wandb_path / "latest-run").iterdir()):
                # Parse out the unique resume_id from the `.wandb` file...
                wandb_fns = [f.name for f in (wandb_path / "latest-run").iterdir() if str(f).endswith(".wandb")]
                assert len(wandb_fns) == 1, f"There should only be 1 `.wandb` file... found {len(wandb_fns)}!"

                # Regex match on `run-{id}.wandb`...
                wandb_resume_id = re.search("run-(.+?).wandb", wandb_fns[0]).group(1)

            # Otherwise, assert that we're starting from scratch!
            else:
                assert start_checkpoint is None, "Trying to restart a run from checkpoint without a valid W&B ID!"

        elif cfg.resume:
            xm.master_print(f"Resuming :: Using Specified W&B Resume ID = `{cfg.wandb_resume_id}`")
            wandb_resume_id = cfg.wandb_resume_id

        # Initialize Weights & Biases
        xm.master_print(f"W&B Resume is {cfg.resume} w/ W&B Resume ID = {wandb_resume_id}!")
        wandb.init(
            project=cfg.tracking.project,
            entity=cfg.tracking.entity,
            config=cfg,
            name=run_id,
            dir=f"{os.getcwd()}" if cfg.tracking.directory is None else cfg.tracking.directory,
            tags=tags,
            notes=cfg.tracking.notes,
            resume="allow" if start_checkpoint is not None else False,
            id=wandb_resume_id,
            # Weird line because PT-TPU VMs don't come with a working install of Tensorflow...
            settings=wandb.Settings(_disable_stats=True),
        )

        # Initialize JSONL Logger (append only mode) --> last "global step" will always take precedence.
        with jsonlines.open(f"{run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
            js_logger.write(
                {
                    "run_id": run_id,
                    "start_time": datetime.now().strftime("%m-%d-%H:%M"),
                    "hparams": OmegaConf.to_container(cfg),
                }
            )

    # Rank Zero Node will take time to spin up the loggers & checkpointer... might as well rendezvous?
    xm.rendezvous("Logging...")

    # === Here Be Dragons ===
    # Time to handle device placement -- Note - example code doesn't specify device idx - why not?
    #   > https://github.com/pytorch/xla/blob/3c0d68da07702995a592ea70f27868cd76fa0755/test/test_train_mp_mnist.py#L114
    #   > Results in printing [xla:0] and [xla:1] a bunch... no [xla:2-7]? This feels bad...?
    #
    #   |=> Debugging Try: `xm.xla_device(n=xm.get_ordinal()) ---> hangs completely?
    #   +=> *ANSWER*: https://github.com/pytorch/xla/issues/2345#issuecomment-657114819
    #       >> "Make no assumptions and don't try to build them manually..."
    device = xm.xla_device()
    model = model.train().to(device)
    optimizer, update_lr = model.configure_optimizer()
    global_step, train_losses, lrs, start_time, resume_time = 0, deque(maxlen=128), [], time.time(), 0

    # If resuming (valid `start_checkpoint`) -- patch the optimizer state dictionary, and load!
    if start_checkpoint is not None:
        patched_optimizer_state_dict = {
            "state": optimizer_state_dict,
            "param_groups": optimizer.state_dict()["param_groups"],
        }
        optimizer.load_state_dict(patched_optimizer_state_dict)

    # Create step timing...
    step_times, step_start_time = deque(maxlen=128), time.time()

    # Create Model/Architecture-Specific Trackers...
    if cfg.model.arch == "v-mvp":
        reconstruction_losses = deque(maxlen=128)

    elif cfg.model.arch in {"v-r3m", "v-rn3m"}:
        tcn_losses, reward_losses, l1_losses, l2_losses = [deque(maxlen=128) for _ in range(4)]
        tcn_accuracies, reward_accuracies = [deque(maxlen=128) for _ in range(2)]

    elif cfg.model.arch == "v-cond":
        reconstruction_losses = deque(maxlen=128)

    elif cfg.model.arch == "v-dual":
        reconstruction_losses = deque(maxlen=128)
        zero_reconstruction, k_reconstruction = deque(maxlen=128), deque(maxlen=128)

    elif cfg.model.arch == "v-gen":
        reconstruction_losses, lm_losses, lm_ppl = deque(maxlen=128), deque(maxlen=128), deque(maxlen=128)
        zero_reconstruction, k_reconstruction = deque(maxlen=128), deque(maxlen=128)

    else:
        raise NotImplementedError(f"Trackers for Model `{cfg.model.arch}` not implemented!")

    # 0th Checkpoint - Pull out optimizer state explicitly (`groups` are not serializable & can easily be replicated)
    saver = XLACheckpointSaver(cfg.tracking.checkpoint_strategy, run_dir, cfg.accelerator.accelerator)
    if start_checkpoint is None and start_epoch == 0:
        xm.master_print("Saving 0th Epoch Checkpoint...")
        saver.save(
            epoch=0, is_local_step=False, model=model, optimizer=optimizer, duration=0, train_loss=None, val_loss=None
        )

    # Run on all processes --> retrieve "0th epoch" dataset!
    #   =>> Important, ensures data is locked across models, for the given epoch!
    xm.master_print(f"Retrieving Dataset `{cfg.dataset.name}` prepared for `{cfg.model.arch}`!")
    train_dataset, val_dataset = get_epoch_datasets(
        0,
        cfg.dataset.name,
        cfg.dataset.normalization,
        cfg.model.arch,
        cfg.dataset.stream,
        cfg.dataset.artifact_path,
        cfg.dataset.stream_prefix,
        cfg.model.data_modality,
        cfg.model.get("lang_dropout", None),
        cfg.model.get("gen_ratio", None),
    )

    # Loading Datasets might take time... rendezvous to be safe
    xm.rendezvous("Retrieved Datasets...")

    # Iterate through Epochs, Evaluating at the end of each Training Epoch!
    #   >> Everything in this loop should happen across all workers, except for the logging (ordinal 0)!
    xm.master_print("Starting Training Loop...")
    for epoch in range(start_epoch, cfg.dataset.max_epochs):
        xm.master_print(f"\t[Epoch {epoch}] Building Distributed Sampler & DataLoaders...")
        train_dataset.set_epoch(epoch)

        # ResumeableDistributedSampler operates at over *examples* --> start_step (full_batches) * effective_bsz
        seen_examples = start_step * cfg.model.effective_bsz
        train_sampler = ResumeableDistributedSampler(
            seen_examples,
            start_epoch,
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
            seed=cfg.seed,
        )

        # Set epoch appropriately for the `train_sampler` --> necessary to trigger "normal" logic!
        train_sampler.set_epoch(epoch)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.model.device_bsz,
            sampler=train_sampler,
            shuffle=False if train_sampler else True,
            num_workers=cfg.accelerator.num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=4,
        )

        # NOTE :: We're not sharding the Validation set --> *everybody* will run forward passes on the *same* data!
        #   > We will have to reduce_mesh() later... unclear why, but the torch_xla folks seem keen on it; might lead to
        #   > weird rendezvous/hang issues if Validation is big enough...
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.model.device_bsz,
            shuffle=False,
            num_workers=4,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

        # Initializing the Dataloaders might take time depending on process...
        xm.rendezvous("Initialized Dataloaders...")

        # Leverage the *special* <XLA ParallelLoader> API that handles synchronizing TPU cores across batches!
        #   > NOTE: This is super important!
        xm.master_print("\tSetting up Parallel MpDeviceLoaders...")
        train_device_loader = parallel.MpDeviceLoader(train_dataloader, device)
        val_device_loader = parallel.MpDeviceLoader(val_dataloader, device)

        # Book-keeping & LR setting when `resuming` --> only do this on start_epoch!
        if epoch == start_epoch:
            if start_checkpoint is not None:
                global_step = start_step + ((len(train_dataset) // cfg.model.effective_bsz) * start_epoch)
                resume_time = int(re.search("-t=(.+?).pt", str(start_checkpoint)).group(1))
                lrs.append(update_lr(start_epoch, start_step / (len(train_dataset) // cfg.model.effective_bsz)))
            else:
                lrs.append(update_lr(start_epoch, 0))

        # Iterate...
        step_start_time = time.time()
        with tqdm(total=len(train_device_loader) // accumulate_grad_batches, disable=not is_rank_zero) as progress:
            for train_idx, batch in enumerate(train_device_loader):
                if cfg.model.arch == "v-mvp":
                    # Run a forward pass through the MAE... other return vals are reconstructions (pixel norm) & mask
                    loss, _, _ = model(batch)
                    reconstruction_losses.append(loss)

                elif cfg.model.arch in {"v-r3m", "v-rn3m"}:
                    imgs, lang, lang_mask = batch
                    loss, tcn_loss, reward_loss, l1_loss, l2_loss, tcn_acc, rew_acc = model(imgs, lang, lang_mask)

                    # Add to trackers
                    tcn_losses.append(tcn_loss)
                    reward_losses.append(reward_loss)
                    l1_losses.append(l1_loss)
                    l2_losses.append(l2_loss)
                    tcn_accuracies.append(tcn_acc)
                    reward_accuracies.append(rew_acc)

                elif cfg.model.arch == "v-cond":
                    img, lang, lang_mask = batch
                    loss, _, _ = model(img, lang, lang_mask)
                    reconstruction_losses.append(loss)

                elif cfg.model.arch == "v-dual":
                    imgs, lang, lang_mask = batch
                    loss, [zero_loss, k_loss] = model(imgs, lang, lang_mask)

                    # Add to trackers
                    reconstruction_losses.append(loss)
                    zero_reconstruction.append(zero_loss)
                    k_reconstruction.append(k_loss)

                elif cfg.model.arch == "v-gen":
                    imgs, lang_con, lang_con_mask, lang_gen, lang_gen_mask, lang_gen_weight = batch
                    loss, reconstruction_loss, lm_loss, [zero_loss, k_loss] = model(
                        imgs, lang_con, lang_con_mask, lang_gen, lang_gen_mask, lang_gen_weight
                    )

                    # Add to trackers
                    reconstruction_losses.append(reconstruction_loss)
                    lm_losses.append(lm_loss)
                    lm_ppl.append(torch.exp(lm_loss))
                    zero_reconstruction.append(zero_loss)
                    k_reconstruction.append(k_loss)

                else:
                    raise NotImplementedError(f"Forward Pass Logic for Model `{cfg.model.arch}` not implemented!")

                # Write Loss to Loggers (prior to accumulation normalization)
                train_losses.append(loss)

                # Normalize loss to account for accumulation
                loss = loss / accumulate_grad_batches
                loss.backward()

                # Gradient Accumulation =>> Note: skip any errant batches at the end...
                if (train_idx + 1) % accumulate_grad_batches == 0:
                    xm.optimizer_step(optimizer)  # Note call to xm.optimizer_step() -- has implicit mark_step!
                    optimizer.zero_grad()

                    # Add to `step_times`
                    step_times.append(time.time() - step_start_time)

                    # Logging --> Because there is no guarantee processes will be in sync, we need a `closure`
                    #   > Ref: https://pytorch.org/xla/release/1.11/index.html#torch_xla.core.xla_model.add_step_closure
                    if is_rank_zero and global_step % cfg.tracking.log_frequency == 0:
                        if cfg.model.arch == "v-mvp":
                            xm.add_step_closure(
                                log_vmvp_train_update,
                                args=(
                                    epoch,
                                    global_step,
                                    run_id,
                                    train_losses,
                                    lrs[-1],
                                    reconstruction_losses,
                                    step_times,
                                ),
                            )

                        elif cfg.model.arch == "v-r3m":
                            xm.add_step_closure(
                                log_vr3m_train_update,
                                args=(
                                    epoch,
                                    global_step,
                                    run_id,
                                    train_losses,
                                    lrs[-1],
                                    tcn_losses,
                                    reward_losses,
                                    l1_losses,
                                    l2_losses,
                                    tcn_accuracies,
                                    reward_accuracies,
                                    step_times,
                                ),
                            )

                        elif cfg.model.arch == "v-rn3m":
                            xm.add_step_closure(
                                log_vrn3m_train_update,
                                args=(
                                    epoch,
                                    global_step,
                                    run_id,
                                    train_losses,
                                    lrs[-1],
                                    tcn_losses,
                                    reward_losses,
                                    l1_losses,
                                    l2_losses,
                                    tcn_accuracies,
                                    reward_accuracies,
                                    step_times,
                                ),
                            )

                        elif cfg.model.arch == "v-cond":
                            xm.add_step_closure(
                                log_vcond_train_update,
                                args=(
                                    epoch,
                                    global_step,
                                    run_id,
                                    train_losses,
                                    lrs[-1],
                                    reconstruction_losses,
                                    step_times,
                                ),
                            )

                        elif cfg.model.arch == "v-dual":
                            xm.add_step_closure(
                                log_vdual_train_update,
                                args=(
                                    epoch,
                                    global_step,
                                    run_id,
                                    train_losses,
                                    lrs[-1],
                                    reconstruction_losses,
                                    zero_reconstruction,
                                    k_reconstruction,
                                    step_times,
                                ),
                            )

                        elif cfg.model.arch == "v-gen":
                            xm.add_step_closure(
                                log_vgen_train_update,
                                args=(
                                    epoch,
                                    global_step,
                                    run_id,
                                    train_losses,
                                    lrs[-1],
                                    reconstruction_losses,
                                    lm_losses,
                                    lm_ppl,
                                    zero_reconstruction,
                                    k_reconstruction,
                                    step_times,
                                ),
                            )

                        else:
                            raise NotImplementedError(f"Log Update for Model `{cfg.model.arch}` not implemented!")

                    # Increment Global Step _after_ logging!
                    global_step += 1

                    # Save checkpoint subject to *local_step = (train_idx + 1) // accumulate_grad_batches*
                    saver.save(
                        epoch=epoch,
                        is_local_step=True,
                        model=model,
                        optimizer=optimizer,
                        duration=int(time.time() - start_time) + resume_time,
                        local_step=start_step + ((train_idx + 1) // accumulate_grad_batches),
                    )

                    # Update LR every `accumulation_steps` iterations...
                    lrs.append(
                        update_lr(
                            epoch,
                            (start_step + ((train_idx + 1) // accumulate_grad_batches))
                            / (len(train_dataset) // cfg.model.effective_bsz),
                        )
                    )

                    # Reset `step_start_time`
                    step_start_time = time.time()

                    # Update `progress` each time we take a gradient step!
                    progress.update()

                # After each forward pass, mark a step, to compile XLA graph for a single forward pass!
                #   =>> Note :: this is important, with gradient accumulation, the graph can get massive otherwise!
                xm.mark_step()

            else:
                # Clear gradients and reset start step (regardless) at end of the loop
                optimizer.zero_grad()
                start_step = 0

        # Redundant, but Synchronous Validation Epoch...
        xm.master_print("Validating...")
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_device_loader, disable=not is_rank_zero):
                if cfg.model.arch == "v-mvp":
                    loss, _, _ = model(batch)
                elif cfg.model.arch in {"v-r3m", "v-rn3m"}:
                    imgs, lang, lang_mask = batch
                    loss, _, _, _, _, _, _ = model(imgs, lang, lang_mask)
                elif cfg.model.arch == "v-cond":
                    img, lang, lang_mask = batch
                    loss, _, _ = model(img, lang, lang_mask)
                elif cfg.model.arch == "v-dual":
                    imgs, lang, lang_mask = batch
                    loss, _ = model(imgs, lang, lang_mask)
                elif cfg.model.arch == "v-gen":
                    imgs, lang_con, lang_con_mask, lang_gen, lang_gen_mask, lang_gen_weight = batch
                    loss, _, _, _ = model(imgs, lang_con, lang_con_mask, lang_gen, lang_gen_mask, lang_gen_weight)
                else:
                    raise NotImplementedError(f"Forward Pass Logic for Model `{cfg.model.arch} not implemented!")

                # Just append to val_losses...
                val_losses.append(loss)

            # Compute Val Loss & *mesh reduce* --> Why? :: the XLA people said so!
            val_loss = torch.stack(val_losses).mean().item()
            val_loss = xm.mesh_reduce("val_loss", val_loss, np.mean)  # All replicas should just return the same thing?

            # Logging --> add another `closure` for end-of-epoch cleanup --> compute `duration` as well...
            duration = int(time.time() - start_time) + resume_time
            if is_rank_zero:
                xm.add_step_closure(
                    log_epoch_end_update,
                    args=(
                        cfg.model.arch,
                        epoch,
                        global_step,
                        run_id,
                        duration,
                        train_losses,
                        val_loss,
                        lrs[-1],
                        step_times,
                    ),
                )

        # Save Checkpoint (at end of Epoch)
        saver.save(
            epoch=epoch + 1,
            is_local_step=False,
            model=model,
            optimizer=optimizer,
            duration=duration,
            train_loss=train_losses[-1].item(),
            val_loss=val_loss,
        )

    # Dump TPU Debugging Metrics...
    if is_rank_zero:
        with open("tpu-debug-metrics.log", "w") as f:
            f.write(met.metrics_report())

    # Exiting w/ Multiprocessing is a Nightmare... try to join?
    xm.master_print("...and that's all, folks!")
    xm.rendezvous("Cheers!")

    # Sleep for like 3 minutes... get W&B to finish syncing logs
    wandb.finish()
    time.sleep(150)


def mp_fn(_: int, cfg: PretrainConfig) -> None:
    torch.set_default_tensor_type("torch.FloatTensor")

    # Let's Start Pretraining!
    xpretrain(cfg)


@hydra.main(config_path=None, config_name="config")
def main(cfg: PretrainConfig) -> None:
    import torch_xla.distributed.xla_multiprocessing as xmp

    # Call XMP Spawn w/ the Config as the sole argument...
    xmp.spawn(mp_fn, args=(cfg,), nprocs=cfg.accelerator.num_accelerators, start_method="spawn")


if __name__ == "__main__":
    main()
