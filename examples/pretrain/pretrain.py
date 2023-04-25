"""
pretrain.py

Core pretraining script for Native PyTorch (Single/Multi-) GPU pretraining on the Something-Something-v2 dataset; this
is basically just a 1-1 reproduction of the XLA pretraining script (`examples/xla-reference/xpretrain.py`) with just
a bit of cleanup, the default PyTorch DDP semantics (`torchrun`), using PyTorch 2.0.

Other notable differences from `xpretrain.py`:
    - Loads data from the local filesystem instead of streaming from a GCP bucket (can be added back easily!)
    - No TPU/XLA specific dependencies --> just PyTorch 2.0!

Run with:
    - [Single Node Multi-GPU ($K)]: `torchrun --standalone --nnodes 1 --nproc-per-node $K examples/pretrain/pretrain.py`
"""
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.distributed as dist
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from voltron.conf import AcceleratorConfig, DatasetConfig, ModelConfig, TrackingConfig
from voltron.datasets import get_datasets
from voltron.models import get_model_optimizer
from voltron.overwatch import OverwatchRich
from voltron.util import CheckpointSaver, Metrics, ResumeableDistributedSampler, do_resume, set_global_seed

# Set Defaults (Hydra w/ Structured Configs)
DEFAULTS = [
    "_self_",
    {"model": "v-cond"},
    {"dataset": "sth-sth-v2"},
    {"accelerator": "torchrun"},
    {"tracking": "voltron-tracking"},
    {"override hydra/job_logging": "overwatch_rich"},
]


@dataclass
class PretrainConfig:
    # fmt: off
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: Dict[str, Any] = field(default_factory=lambda: {
        "run": {"dir": "runs/train/${model.identifier}+dataset-${dataset.name}"}
    })

    # Command Line Arguments
    run_id: Optional[str] = None                                        # Run ID for Logging
    seed: int = 21                                                      # Random Seed (for reproducibility)

    # Resume / Debug Behavior
    resume: bool = True                                                 # Whether to resume an existing run...
    wandb_resume_id: Optional[str] = None                               # W&B Run ID for `resume` behavior...

    # Composable / Structured Arguments
    model: ModelConfig = MISSING                                        # Model architecture for pretraining
    dataset: DatasetConfig = MISSING                                    # List of datasets for pretraining
    accelerator: AcceleratorConfig = MISSING                            # Accelerator (should always keep `torchrun`)
    tracking: TrackingConfig = MISSING                                  # Run/experiment tracking configuration
    # fmt: on


# Hydra Setup :: Retrieve ConfigStore (Singleton) & Register Components
cs = ConfigStore.instance()
cs.store(group="hydra/job_logging", name="overwatch_rich", node=OverwatchRich)
cs.store(name="config", node=PretrainConfig)


@hydra.main(config_path=None, config_name="config")
def pretrain(cfg: PretrainConfig) -> None:
    # Initialize Distributed Process Group --> assumes NCCL + Environment Variable Initialization (via `torchrun`)
    dist.init_process_group(backend="nccl", init_method="env://")
    device_id = dist.get_rank() % torch.cuda.device_count()
    is_rank_zero, rank, world_size = dist.get_rank() == 0, dist.get_rank(), dist.get_world_size()

    # Create Unique Run Name -- `resume = True` we assume the same "run_id"
    if cfg.run_id is None:
        cfg.run_id = run_dir = f"{cfg.model.identifier}+{cfg.dataset.name}-ddp-x{cfg.seed}"
    else:
        run_dir = cfg.run_id

    # Setup Logging (Rank 0 Only!) and Directory Handling
    overwatch = logging.getLogger(__file__)
    overwatch.setLevel(logging.INFO if is_rank_zero else logging.ERROR)
    overwatch.info("Voltron Training :: Assembling the Legendary Defender...")
    if is_rank_zero:
        os.makedirs(run_dir, exist_ok=True)

    # Let's Get Started!
    overwatch.info(
        '\t=>> "If you get too worried about what could go wrong, you might miss a chance to do something great."'
    )

    # Set Randomness & Get Dataloader `worker_init_fn` to ensure proper randomness in augmentations (if any)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    # Initialize Model & Optimizer --> Wrap in DDP / Device Handling
    #   > Note :: For (Standard) DDP Training --> initializing Optimizer before DDP == initializing after!
    overwatch.info("Initializing Model, Optimizer, and Learning Rate Scheduler")
    model, optimizer, update_lr = get_model_optimizer(cfg.model, cfg.dataset)
    model = DDP(model.to(device_id), device_ids=[device_id], output_device=device_id)

    # Handle Resume / Checkpoint Loading
    resume_checkpoint, resume_epoch, resume_step = do_resume(cfg.resume, run_dir=run_dir)
    if resume_checkpoint is not None:
        # IMPORTANT --> Load weights by mapping specifically to `cuda:<device_id>`!
        resume_state = torch.load(resume_checkpoint, map_location=f"cuda:{device_id}")
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])

        dist.barrier()

    # Create Checkpoint Saver and Save Initial Checkpoint
    saver = CheckpointSaver(cfg.tracking.checkpoint_strategy, run_dir, is_rank_zero=is_rank_zero)
    if resume_checkpoint is None and resume_epoch == 0:
        overwatch.info("  | Saving 0th Epoch Checkpoint (Model Initialization)")
        saver.save(
            epoch=0, is_local_step=False, model=model, optimizer=optimizer, duration=0, train_loss=None, val_loss=None
        )
        dist.barrier()

    # Get Datasets --> Barrier after I/O Intensive Operation
    overwatch.info(f"Retrieving Dataset `{cfg.dataset.name}` prepared for Model `{cfg.model.arch}`")
    train_dataset, val_dataset = get_datasets(
        0,
        cfg.dataset.name,
        cfg.model.arch,
        cfg.dataset.artifact_path,
        cfg.model.data_modality,
        cfg.dataset.resolution,
        cfg.dataset.normalization,
        cfg.model.get("lang_dropout", None),
        cfg.model.get("gen_ratio", None),
    )
    dist.barrier()

    # Create Metrics =>> Handles on-the-fly computation, logging to JSONL and Weights & Biases
    metrics = Metrics(
        active_loggers=cfg.tracking.active_loggers,
        run_id=cfg.run_id,
        hparams=OmegaConf.to_container(cfg),
        model_arch=cfg.model.arch,
        is_rank_zero=is_rank_zero,
        tracking_cfg=cfg.tracking,
        tags=cfg.tracking.tags,
        resume=cfg.resume,
        resume_id=cfg.wandb_resume_id,
    )
    dist.barrier()

    # Configure Gradient Accumulation --> function of `effective_bsz`, `native_bsz`, and `WORLD_SIZE`
    assert cfg.model.effective_bsz % cfg.model.native_bsz == 0, "Device `native_bsz` must evenly divide `effective_bsz`"
    accumulate_grad_batches = cfg.model.effective_bsz // cfg.model.native_bsz // world_size
    overwatch.info(f"Running `{cfg.model.identifier}` Model Pretraining with Parameters =>")
    overwatch.info(f"  | Effective Batch Size = `{cfg.model.effective_bsz}`")
    overwatch.info(f"  | Per-Device Batch Size = `{cfg.model.native_bsz}`")
    overwatch.info(f"  | Distributed World Size = `{world_size}`")
    overwatch.info(f"  | Accumulation Steps = `{accumulate_grad_batches}`")

    # Start Train Loop --> Iterate through Epochs (Evaluation at end of Epoch)
    overwatch.info("Starting Training Loop")
    for epoch in range(resume_epoch, cfg.dataset.max_epochs):
        overwatch.info(f"  | [Epoch {epoch:03d}] Building Distributed Sampler & DataLoaders")
        train_dataset.set_epoch(epoch)
        dist.barrier()

        # [Custom] ResumeableDistributedSampler operates over *examples* --> start_step (full batches) * effective_bsz
        train_sampler = ResumeableDistributedSampler(
            seen_examples=resume_step * cfg.model.effective_bsz,
            resume_epoch=resume_epoch,
            dataset=train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.seed,
        )
        train_sampler.set_epoch(epoch)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

        # Create Epoch DataLoaders
        train_dl = DataLoader(
            train_dataset,
            batch_size=cfg.model.native_bsz,
            sampler=train_sampler,
            shuffle=False,
            num_workers=cfg.accelerator.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=4,
            worker_init_fn=worker_init_fn,
        )
        val_dl = DataLoader(
            val_dataset, batch_size=cfg.model.native_bsz, sampler=val_sampler, shuffle=False, num_workers=4
        )

        # Book-Keeping =>> Set LR when `resume = True` (or starting from scratch)
        if epoch == resume_epoch or epoch == 0:
            metrics.resume_time = (
                int(re.search("-t=(.+?).pt", str(resume_checkpoint)).group(1)) if resume_checkpoint is not None else 0
            )
            metrics.commit(
                global_step=resume_step + ((len(train_dataset) // cfg.model.effective_bsz) * resume_epoch),
                lr=update_lr(resume_epoch, resume_step / (len(train_dataset) // cfg.model.effective_bsz)),
                update_step_time=True,
            )

        # === Train Epoch ===
        model.train()
        status = metrics.get_status(epoch)
        overwatch.info(f"  | [Epoch {epoch:03d}] Running Train Loop")
        with tqdm(
            total=len(train_dl) // accumulate_grad_batches, desc=status, leave=False, disable=not is_rank_zero
        ) as progress:
            for train_idx, batch in enumerate(train_dl):
                # Model-Specific Handling
                if cfg.model.arch == "v-mvp":
                    img = batch
                    loss, _, _ = model(img.to(device_id, non_blocking=True))
                    metrics.commit(reconstruction_loss=loss)

                elif cfg.model.arch in {"v-r3m", "v-rn3m"}:
                    imgs, lang, lang_mask = batch
                    loss, tcn_loss, reward_loss, l1_loss, l2_loss, tcn_acc, rew_acc = model(
                        imgs.to(device_id, non_blocking=True),
                        lang.to(device_id, non_blocking=True),
                        lang_mask.to(device_id, non_blocking=True),
                    )
                    metrics.commit(
                        tcn_loss=tcn_loss,
                        reward_loss=reward_loss,
                        l1_loss=l1_loss,
                        l2_loss=l2_loss,
                        tcn_accuracy=tcn_acc,
                        reward_accuracy=rew_acc,
                    )

                elif cfg.model.arch == "v-cond":
                    img, lang, lang_mask = batch
                    loss, _, _ = model(
                        img.to(device_id, non_blocking=True),
                        lang.to(device_id, non_blocking=True),
                        lang_mask.to(device_id, non_blocking=True),
                    )
                    metrics.commit(reconstruction_loss=loss)

                elif cfg.model.arch == "v-dual":
                    imgs, lang, lang_mask = batch
                    loss, [zero_loss, k_loss] = model(
                        imgs.to(device_id, non_blocking=True),
                        lang.to(device_id, non_blocking=True),
                        lang_mask.to(device_id, non_blocking=True),
                    )
                    metrics.commit(
                        reconstruction_loss=loss,
                        zero_reconstruction_loss=zero_loss,
                        k_reconstruction_loss=k_loss,
                    )

                elif cfg.model.arch == "v-gen":
                    imgs, lang_con, lang_con_mask, lang_gen, lang_gen_mask, lang_gen_weight = batch
                    loss, reconstruction_loss, lm_loss, [zero_loss, k_loss] = model(
                        imgs.to(device_id, non_blocking=True),
                        lang_con.to(device_id, non_blocking=True),
                        lang_con_mask.to(device_id, non_blocking=True),
                        lang_gen.to(device_id, non_blocking=True),
                        lang_gen_mask.to(device_id, non_blocking=True),
                        lang_gen_weight,
                    )
                    metrics.commit(
                        reconstruction_loss=reconstruction_loss,
                        zero_reconstruction_loss=zero_loss,
                        k_reconstruction_loss=k_loss,
                        lm_loss=lm_loss,
                        lm_ppl=torch.exp(lm_loss),
                    )

                else:
                    raise ValueError(f"Forward() for Model `{cfg.model.arch}` is not implemented!")

                # Commit Loss (Prior to Normalization)
                metrics.commit(loss=loss)

                # Normalize Loss to account for Gradient Accumulation --> Backward!
                normalized_loss = loss / accumulate_grad_batches
                normalized_loss.backward()

                # Step =>> Check if done w/ Gradient Accumulation
                if (train_idx + 1) % accumulate_grad_batches == 0:
                    metrics.commit(update_step_time=True)

                    # Push Metrics every `log_frequency` steps...
                    if metrics.global_step % cfg.tracking.log_frequency == 0:
                        status = metrics.push(epoch)

                    # Optimizer Step --> Increment Global Step, Learning Rate, and Checkpoint (if specified)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr = update_lr(
                        epoch,
                        (resume_step + ((train_idx + 1) // accumulate_grad_batches))
                        / (len(train_dataset) // cfg.model.effective_bsz),
                    )
                    metrics.commit(global_step=metrics.global_step + 1, lr=lr)
                    saver.save(
                        epoch,
                        is_local_step=True,
                        model=model,
                        optimizer=optimizer,
                        duration=int(time.time() - metrics.start_time) + metrics.resume_time,
                        local_step=resume_step + ((train_idx + 1) // accumulate_grad_batches),
                    )

                    # Update Progress Bar
                    progress.update()
                    progress.set_description(status)

        # === After Train Epoch --> Clear Gradients and reset `resume_step` ===
        optimizer.zero_grad()
        resume_step = 0

        # === Validation ===
        overwatch.info(f"  | [Epoch {epoch:03d}] Running Validation Loop")
        model.eval()

        # Accumulate `validation_losses` in order to `all_reduce` later!
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_dl, disable=not is_rank_zero, leave=False):
                # Model-Specific Handling
                if cfg.model.arch == "v-mvp":
                    img = batch
                    val_loss, _, _ = model(img.to(device_id, non_blocking=True))

                elif cfg.model.arch in {"v-r3m", "v-rn3m"}:
                    imgs, lang, lang_mask = batch
                    val_loss, _, _, _, _, _, _ = model(
                        imgs.to(device_id, non_blocking=True),
                        lang.to(device_id, non_blocking=True),
                        lang_mask.to(device_id, non_blocking=True),
                    )

                elif cfg.model.arch == "v-cond":
                    img, lang, lang_mask = batch
                    val_loss, _, _ = model(
                        img.to(device_id, non_blocking=True),
                        lang.to(device_id, non_blocking=True),
                        lang_mask.to(device_id, non_blocking=True),
                    )

                elif cfg.model.arch == "v-dual":
                    imgs, lang, lang_mask = batch
                    val_loss, _ = model(
                        imgs.to(device_id, non_blocking=True),
                        lang.to(device_id, non_blocking=True),
                        lang_mask.to(device_id, non_blocking=True),
                    )

                elif cfg.model.arch == "v-gen":
                    imgs, lang_con, lang_con_mask, lang_gen, lang_gen_mask, lang_gen_weight = batch
                    val_loss, _, _, _ = model(
                        imgs.to(device_id, non_blocking=True),
                        lang_con.to(device_id, non_blocking=True),
                        lang_con_mask.to(device_id, non_blocking=True),
                        lang_gen.to(device_id, non_blocking=True),
                        lang_gen_mask.to(device_id, non_blocking=True),
                        lang_gen_weight,
                    )

                else:
                    raise ValueError(f"Forward() for Model `{cfg.model.arch}` is not implemented!")

                # Add to Validation Losses
                val_losses.append(val_loss)

        # All Reduce --> Push Epoch Metrics --> Checkpoint!
        validation_loss = torch.stack(val_losses).mean()
        dist.all_reduce(validation_loss)
        avg_val_loss = validation_loss / world_size
        if is_rank_zero:
            epoch_status, train_loss, training_duration = metrics.push_epoch(epoch, avg_val_loss)
            saver.save(
                epoch=epoch + 1,
                is_local_step=False,
                model=model,
                optimizer=optimizer,
                duration=training_duration,
                train_loss=train_loss.item(),
                val_loss=avg_val_loss.item(),
            )

        # === End of Epoch ===
        dist.barrier()

    # Finalize
    metrics.finalize()

    # And... we're done!
    overwatch.info("...and that's all, folks!")
    dist.barrier()


if __name__ == "__main__":
    # General Defaults --> should use Tensor Cores (kinda) if you have them!
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_start_method("spawn", force=True)

    pretrain()
