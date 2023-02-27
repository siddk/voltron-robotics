"""
vmvp.py

PyTorch Module defining a basic MAE a la Masked Visual Pretraining for Motor Control (MVP), with the requisite
hyperparameters - as defined in the original ImageMAE paper, and as used by both MVP papers.

References:
    - https://github.com/facebookresearch/mae
    - https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from voltron.models.util.optimization import get_lr_update
from voltron.models.util.transformer import Block, PatchEmbed, get_2D_position_embeddings


class VMVP(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        encoder_depth: int,
        encoder_embed_dim: int,
        encoder_n_heads: int,
        decoder_depth: int,
        decoder_embed_dim: int,
        decoder_n_heads: int,
        optimizer: str,
        schedule: str,
        base_lr: float,
        min_lr: float,
        effective_bsz: float,
        betas: Tuple[float, float],
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        mask_ratio: float = 0.75,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        norm_pixel_loss: bool = True,
    ):
        """
        Initialize an VMVP (MAE) model with the requisite architecture parameters.

        :param resolution: Base image resolution -- usually 224 (ImageNet size).
        :param patch_size: Height/Width of each patch in pixels -- usually 16.
        :param encoder_depth: Number of Transformer blocks in the encoder -- should be greater than decoder.
        :param encoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param encoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param decoder_depth: Number of Transformer blocks in the decoder -- should be relatively shallow.
        :param decoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param decoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param optimizer: String denoting which optimizer to use (for MAEs, usually `adamw`)
        :param schedule: Learning rate schedule to use; for Transformers a linear warmup + decay is recommended!
        :param base_lr: Base learning rate, to be scaled via a linear scaling rule (from scaling laws).
        :param min_lr: Minimum learning rate to decay to over the course of learning (usually 0.0)
        :param effective_bsz: Global batch size for update, dictates the scaling of the base_lr.
        :param betas: Adam optimizer betas (only applicable for `adam` and `adamw`. Prevents early loss spiking.
        :param weight_decay: Weight decay for global weight regularization (only applied to non-bias, non-LN layers).
        :param warmup_epochs: Number of epochs to warmup learning rate for linear warmup schedule.
        :param max_epochs: Total number of training epochs to be run.
        :param mask_ratio: Ratio for number of patches to mask out for MAE -- should be fairly high!
        :param mlp_ratio: Ratio for embedding size to Position-wise FeedForward MLP (gets shrunk back down).
        :param in_channels: Default number of channels in the base image -- almost always 3.
        :param norm_pixel_loss: Normalize decoder pixel targets for reconstruction (better perf, not interpretable).
        """
        super().__init__()
        self.resolution, self.patch_size, self.mask_ratio = resolution, patch_size, mask_ratio
        self.in_channels, self.norm_pixel_loss, self.mlp_ratio = in_channels, norm_pixel_loss, mlp_ratio
        self.optimizer, self.schedule, self.betas, self.weight_decay = optimizer, schedule, betas, weight_decay
        self.lr, self.base_lr, self.min_lr, self.effective_bsz = None, base_lr, min_lr, effective_bsz
        self.warmup_epochs, self.max_epochs = warmup_epochs, max_epochs

        # Encoder/Decoder Parameters
        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.encoder_embed_dim, self.encoder_n_heads = encoder_embed_dim, encoder_n_heads
        self.decoder_embed_dim, self.decoder_n_heads = decoder_embed_dim, decoder_n_heads

        # MAE Encoder Parameters --> MVP uses a CLS Token for feature extraction!
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.patch2embed = PatchEmbed(
            self.resolution, self.patch_size, self.encoder_embed_dim, in_channels=self.in_channels
        )
        self.encoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + 1, self.encoder_embed_dim), requires_grad=False
        )
        self.encoder_blocks = nn.ModuleList(
            [Block(self.encoder_embed_dim, self.encoder_n_heads, self.mlp_ratio) for _ in range(self.encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(self.encoder_embed_dim, eps=1e-6)

        # Projection from Encoder to Decoder
        self.encoder2decoder = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)

        # MAE Decoder Parameters -- Remember the CLS Token!
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + 1, self.decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(self.decoder_embed_dim, self.decoder_n_heads, self.mlp_ratio) for _ in range(self.decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim, eps=1e-6)
        self.decoder_prediction = nn.Linear(self.decoder_embed_dim, (patch_size**2) * in_channels, bias=True)

        # Initialize all Weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Position Encoding -- Fixed 2D Sine-Cosine Embeddings
        enc_pe = get_2D_position_embeddings(self.encoder_embed_dim, int(self.patch2embed.num_patches**0.5), True)
        self.encoder_pe.data.copy_(torch.from_numpy(enc_pe).float().unsqueeze(0))
        dec_pe = get_2D_position_embeddings(self.decoder_embed_dim, int(self.patch2embed.num_patches**0.5), True)
        self.decoder_pe.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))

        # Initialize PatchEmbedding as a Linear...
        nn.init.xavier_uniform_(self.patch2embed.proj.weight.data.view([self.patch2embed.proj.weight.data.shape[0], -1]))

        # Initialize CLS Token & Mask Token w/ Truncated Normal
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Everything else...
        self.apply(self.transformer_initializer)

    @staticmethod
    def transformer_initializer(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # Use xavier_uniform following Jax ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def mask(
        self, patches: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-sample random masking by shuffling :: uses argsort random noise to identify masked patches"""
        bsz, n_patches, embed_dim = patches.shape
        if mask_ratio is not None:
            n_keep = int(n_patches * (1 - mask_ratio))
        else:
            n_keep = int(n_patches * (1 - self.mask_ratio))

        # Sample some noise of n_patches size, argsort to get shuffled IDs (keep small), argsort again to get "unshuffle"
        #   > For clarity -- argsort is an invertible transformation (if argsort `restore`, recovers `shuffle`)
        shuffle_idxs = torch.argsort(torch.rand(bsz, n_patches, device=patches.device), dim=1)
        restore_idxs = torch.argsort(shuffle_idxs, dim=1)

        # Get "keep" (visible) patches
        visible_patches = torch.gather(patches, dim=1, index=shuffle_idxs[:, :n_keep, None].repeat(1, 1, embed_dim))

        # Generate the binary mask --> IMPORTANT :: `0` is keep, `1` is remove (following FAIR MAE convention)
        mask = torch.ones(bsz, n_patches, device=patches.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=restore_idxs)

        return visible_patches, mask, restore_idxs

    def get_representations(self, img: torch.Tensor, mode: str = "patch") -> torch.Tensor:
        """
        Given a single image, extract representations subject to the specified mode in < patch | cls >, where "cls"
        denotes extracting the <CLS> token embedding; for our experiments, we find that running multiheaded attention
        pooling on top of the "patch" embeddings is *always* better!

        :param img: Processed batch of images :: [bsz, 3, 224, 224]
        :param mode: Type of representation to extract -- `patch` (sequence of patch embeddings) or `cls` (<CLS>)

        :return: Extracted representations given img input.
        """
        assert img.ndim == 4, "Invalid input to `get_representations()`"
        assert mode in {"patch", "cls"}, f"Extraction mode `{mode}` not supported!"

        # Extract desired representations
        representations = self.encode(img)
        return representations[:, 1:] if mode == "patch" else representations[:, :1]

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Run a single image through the MAE and extract patch embeddings."""

        # Note: All of this code is taken near-verbatim from the MVP repository...
        #   > Ref: https://github.com/ir413/mvp/blob/master/mvp/backbones/vit.py#L30
        patches = self.patch2embed(img)
        patches_pe = patches + self.encoder_pe[:, 1:, :]

        # Add CLS Token
        cls_token = self.cls_token + self.encoder_pe[:, :1, :]
        cls_tokens = cls_token.expand(img.shape[0], -1, -1)
        cls_patches = torch.cat([cls_tokens, patches_pe], dim=1)

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            cls_patches = block(cls_patches)
        cls_patches = self.encoder_norm(cls_patches)
        return cls_patches

    def forward_encoder(
        self, imgs: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Patchify + Position Embedding (without the CLS Token)
        patches = self.patch2embed(imgs)
        patches_pe = patches + self.encoder_pe[:, 1:, :]

        # Create mask (and go ahead and mask out patches at the same time)
        visible_patches, mask, restore_idxs = self.mask(patches_pe, mask_ratio)

        # Add the CLS Token
        cls_token = self.cls_token + self.encoder_pe[:, :1, :]
        cls_tokens = cls_token.expand(imgs.shape[0], -1, -1)
        cls_visible_patches = torch.cat([cls_tokens, visible_patches], dim=1)

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            cls_visible_patches = block(cls_visible_patches)
        cls_visible_patches = self.encoder_norm(cls_visible_patches)

        return cls_visible_patches, mask, restore_idxs

    def forward_decoder(self, visible_patches: torch.Tensor, restore_idxs: torch.Tensor) -> torch.Tensor:
        # Project patches into decoder embedding dimension
        projected_patches = self.encoder2decoder(visible_patches)

        # Add Mask Tokens to Sequence
        mask_tokens = self.mask_token.repeat(
            projected_patches.shape[0], restore_idxs.shape[1] - visible_patches.shape[1], 1
        )

        # Remove & add back CLS Token as part of the "unshuffling"
        concatenated_patches = torch.cat([projected_patches[:, 1:, :], mask_tokens], dim=1)  # Skip CLS Token
        unshuffled_patches = torch.gather(
            concatenated_patches, dim=1, index=restore_idxs[..., None].repeat(1, 1, projected_patches.shape[2])
        )
        cls_unshuffled_patches = torch.cat([projected_patches[:, :1, :], unshuffled_patches], dim=1)  # Add CLS Token

        # Add Position Embeddings
        cls_decoder_patches = cls_unshuffled_patches + self.decoder_pe

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            cls_decoder_patches = block(cls_decoder_patches)
        cls_decoder_patches = self.decoder_norm(cls_decoder_patches)

        # Run final projection, remove the CLS token, and return
        cls_decoder_prediction = self.decoder_prediction(cls_decoder_patches)
        decoder_prediction = cls_decoder_prediction[:, 1:, :]
        return decoder_prediction

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images to their patched equivalents, by naive reshaping"""
        return rearrange(
            imgs,
            "bsz c (height patch_h) (width patch_w) -> bsz (height width) (patch_h patch_w c)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )

    def compute_loss(self, imgs: torch.Tensor, reconstructions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self.norm_pixel_loss, "`norm_pixel_loss` should always be true... false only for visualizations!"
        targets = self.patchify(imgs)

        # Normalize targets...
        mu, var = targets.mean(dim=-1, keepdim=True), targets.var(dim=-1, unbiased=True, keepdim=True)
        targets = (targets - mu) / ((var + 1e-6) ** 0.5)

        # Compute mean loss per patch first...
        mse = (reconstructions - targets) ** 2
        avg_loss_per_patch = mse.mean(dim=-1)

        # Compute mean loss only on *removed* patches and return
        return (avg_loss_per_patch * mask).sum() / mask.sum()

    def forward(
        self, imgs: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visible_patches, mask, restore_idxs = self.forward_encoder(imgs, mask_ratio)
        reconstructions = self.forward_decoder(visible_patches, restore_idxs)
        loss = self.compute_loss(imgs, reconstructions, mask)

        return loss, reconstructions, mask

    def configure_optimizer(self) -> Tuple[torch.optim.Optimizer, Callable[[int, float], float]]:
        # Short-Circuit on Valid Optimizers
        if self.optimizer not in ["adamw"]:
            raise NotImplementedError(f"Optimizer `{self.optimizer}` not supported - try [`adamw`] instead!")

        # Create Parameter Groups --> Bias terms, Normalization layer parameters shouldn't be decayed...
        #   > This is a compact rewrite of `param_groups_weight_decay()` from TIMM because I don't want the dependency
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check on any parameters with fewer than 2 dimensions or with "bias" in the name...
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        # Build Parameter Groups
        groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

        # Compute LR -- MAE uses the `linear scaling rule` :: lr = base_lr * (effective_bsz / 256)
        #   > https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md
        self.lr = self.base_lr * (self.effective_bsz / 256)

        # Create Optimizer & LR Scheduler
        optimizer = torch.optim.AdamW(groups, lr=self.lr, betas=self.betas)
        update_lr = get_lr_update(optimizer, self.schedule, self.lr, self.min_lr, self.warmup_epochs, self.max_epochs)
        return optimizer, update_lr
