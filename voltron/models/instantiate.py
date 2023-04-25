"""
instantiate.py

Simple wrapping script for instantiating a core Voltron/reproduction model and configuring the torch.Optimizer for DDP
pretraining. Meant to be modular and extensible!
"""
from typing import Callable, Tuple

import torch.nn as nn
from torch.optim import Optimizer

from voltron.conf import DatasetConfig, ModelConfig

from .core.vcond import VCond
from .core.vdual import VDual
from .core.vgen import VGen
from .reproductions.vmvp import VMVP
from .reproductions.vr3m import VR3M
from .reproductions.vrn3m import VRN3M


def get_model_optimizer(
    model_cfg: ModelConfig, dataset_cfg: DatasetConfig
) -> Tuple[nn.Module, Optimizer, Callable[[int, float], float]]:
    """Switch on `model_cfg.arch` --> instantiate the correct nn.Module and Optimizer (on CPU/default device)."""

    # Data-Locked Reproductions
    if model_cfg.arch == "v-mvp":
        model = VMVP(
            resolution=dataset_cfg.resolution,
            patch_size=model_cfg.patch_size,
            encoder_depth=model_cfg.encoder_depth,
            encoder_embed_dim=model_cfg.encoder_embed_dim,
            encoder_n_heads=model_cfg.encoder_n_heads,
            decoder_depth=model_cfg.decoder_depth,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_n_heads=model_cfg.decoder_n_heads,
            optimizer=model_cfg.optimizer,
            schedule=model_cfg.schedule,
            base_lr=model_cfg.base_lr,
            min_lr=model_cfg.min_lr,
            effective_bsz=model_cfg.effective_bsz,
            betas=model_cfg.betas,
            weight_decay=model_cfg.weight_decay,
            warmup_epochs=dataset_cfg.warmup_epochs,
            max_epochs=dataset_cfg.max_epochs,
            mlp_ratio=model_cfg.mlp_ratio,
            norm_pixel_loss=model_cfg.norm_pixel_loss,
        )

    elif model_cfg.arch == "v-r3m":
        model = VR3M(
            resolution=dataset_cfg.resolution,
            patch_size=model_cfg.patch_size,
            depth=model_cfg.depth,
            embed_dim=model_cfg.embed_dim,
            n_heads=model_cfg.n_heads,
            language_model=model_cfg.language_model,
            hf_cache=model_cfg.hf_cache,
            language_dim=model_cfg.language_dim,
            reward_dim=model_cfg.reward_dim,
            n_negatives=model_cfg.n_negatives,
            lang_reward_weight=model_cfg.lang_reward_weight,
            tcn_weight=model_cfg.tcn_weight,
            l1_weight=model_cfg.l1_weight,
            l2_weight=model_cfg.l2_weight,
            optimizer=model_cfg.optimizer,
            schedule=model_cfg.schedule,
            lr=model_cfg.lr,
            min_lr=model_cfg.min_lr,
            warmup_epochs=dataset_cfg.warmup_epochs,
            max_epochs=dataset_cfg.max_epochs,
            mlp_ratio=model_cfg.mlp_ratio,
        )

    elif model_cfg.arch == "v-rn3m":
        model = VRN3M(
            resolution=dataset_cfg.resolution,
            fc_dim=model_cfg.fc_dim,
            language_model=model_cfg.language_model,
            hf_cache=model_cfg.hf_cache,
            language_dim=model_cfg.language_dim,
            reward_dim=model_cfg.reward_dim,
            n_negatives=model_cfg.n_negatives,
            lang_reward_weight=model_cfg.lang_reward_weight,
            tcn_weight=model_cfg.tcn_weight,
            l1_weight=model_cfg.l1_weight,
            l2_weight=model_cfg.l2_weight,
            optimizer=model_cfg.optimizer,
            lr=model_cfg.lr,
        )

    # Voltron Models
    elif model_cfg.arch == "v-cond":
        model = VCond(
            resolution=dataset_cfg.resolution,
            patch_size=model_cfg.patch_size,
            encoder_depth=model_cfg.encoder_depth,
            encoder_embed_dim=model_cfg.encoder_embed_dim,
            encoder_n_heads=model_cfg.encoder_n_heads,
            decoder_depth=model_cfg.decoder_depth,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_n_heads=model_cfg.decoder_n_heads,
            language_model=model_cfg.language_model,
            hf_cache=model_cfg.hf_cache,
            language_dim=model_cfg.language_dim,
            optimizer=model_cfg.optimizer,
            schedule=model_cfg.schedule,
            base_lr=model_cfg.base_lr,
            min_lr=model_cfg.min_lr,
            effective_bsz=model_cfg.effective_bsz,
            betas=model_cfg.betas,
            weight_decay=model_cfg.weight_decay,
            warmup_epochs=dataset_cfg.warmup_epochs,
            max_epochs=dataset_cfg.max_epochs,
            mlp_ratio=model_cfg.mlp_ratio,
            norm_pixel_loss=model_cfg.norm_pixel_loss,
        )

    elif model_cfg.arch == "v-dual":
        model = VDual(
            resolution=dataset_cfg.resolution,
            patch_size=model_cfg.patch_size,
            encoder_depth=model_cfg.encoder_depth,
            encoder_embed_dim=model_cfg.encoder_embed_dim,
            encoder_n_heads=model_cfg.encoder_n_heads,
            decoder_depth=model_cfg.decoder_depth,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_n_heads=model_cfg.decoder_n_heads,
            language_model=model_cfg.language_model,
            hf_cache=model_cfg.hf_cache,
            language_dim=model_cfg.language_dim,
            optimizer=model_cfg.optimizer,
            schedule=model_cfg.schedule,
            base_lr=model_cfg.base_lr,
            min_lr=model_cfg.min_lr,
            effective_bsz=model_cfg.effective_bsz,
            betas=model_cfg.betas,
            weight_decay=model_cfg.weight_decay,
            warmup_epochs=dataset_cfg.warmup_epochs,
            max_epochs=dataset_cfg.max_epochs,
            mlp_ratio=model_cfg.mlp_ratio,
            norm_pixel_loss=model_cfg.norm_pixel_loss,
        )

    elif model_cfg.arch == "v-gen":
        model = VGen(
            resolution=dataset_cfg.resolution,
            patch_size=model_cfg.patch_size,
            encoder_depth=model_cfg.encoder_depth,
            encoder_embed_dim=model_cfg.encoder_embed_dim,
            encoder_n_heads=model_cfg.encoder_n_heads,
            decoder_depth=model_cfg.decoder_depth,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_n_heads=model_cfg.decoder_n_heads,
            language_model=model_cfg.language_model,
            hf_cache=model_cfg.hf_cache,
            language_dim=model_cfg.language_dim,
            max_lang_len=dataset_cfg.max_lang_len,
            vocab_size=model_cfg.vocab_size,
            mae_weight=model_cfg.mae_weight,
            lm_weight=model_cfg.lm_weight,
            optimizer=model_cfg.optimizer,
            schedule=model_cfg.schedule,
            base_lr=model_cfg.base_lr,
            min_lr=model_cfg.min_lr,
            effective_bsz=model_cfg.effective_bsz,
            betas=model_cfg.betas,
            weight_decay=model_cfg.weight_decay,
            warmup_epochs=dataset_cfg.warmup_epochs,
            max_epochs=dataset_cfg.max_epochs,
            mlp_ratio=model_cfg.mlp_ratio,
            norm_pixel_loss=model_cfg.norm_pixel_loss,
        )

    else:
        raise ValueError(f"Model Architecture `{model_cfg.arch}` is not implemented!")

    # Configure Optimizer --> on same device (CPU)
    optimizer, update_lr = model.configure_optimizer()

    return model, optimizer, update_lr
