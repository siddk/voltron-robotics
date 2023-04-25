"""
vdual.py

PyTorch Module defining the Voltron `V-Dual` variant (dual-frame with language-conditioning). In general, follows the
MAE recipe, with the same modifications described in `vcond.py`.

When masking visual patches, the same patches are elided for both the 0th frame and the Kth frame to avoid cheating!

References:
    - https://github.com/huggingface/m4/blob/main/m4/modeling/pretraining/video/videomae.py
    - https://github.com/lucidrains/vit-pytorch
    - https://github.com/MCG-NJU/VideoMAE
"""
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from einops import rearrange, repeat

from voltron.models.util.optimization import get_lr_update
from voltron.models.util.transformer import Block, PatchEmbed, RMSNorm, get_2D_position_embeddings

# Suppress Transformers Logging
transformers.logging.set_verbosity_error()


class VDual(nn.Module):
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
        language_model: str,
        hf_cache: str,
        language_dim: int,
        optimizer: str,
        schedule: str,
        base_lr: float,
        min_lr: float,
        effective_bsz: int,
        betas: Tuple[float, float],
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        mask_ratio: float = 0.75,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        norm_pixel_loss: bool = True,
        use_cls_token: bool = False,
    ) -> None:
        """
        Initialize a VDual model with the requisite architecture parameters.

        :param resolution: Base image resolution -- usually 224 (ImageNet size).
        :param patch_size: Height/Width of each patch in pixels -- usually 16.
        :param encoder_depth: Number of Transformer blocks in the encoder -- should be greater than decoder.
        :param encoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param encoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param decoder_depth: Number of Transformer blocks in the decoder -- should be relatively shallow.
        :param decoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param decoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param language_model: Language model to freeze for encoding narrations/utterances.
        :param hf_cache: Cache directory to store pretrained models, for safe distributed training.
        :param language_dim: Dimensionality of the language embedding coming out of the pretrained LM.
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
        :param use_cls_token: Add <CLS> token for continued pretraining (NOTE: not used in MAE pretraining/finetuning!)
        """
        super().__init__()
        self.resolution, self.patch_size, self.mask_ratio = resolution, patch_size, mask_ratio
        self.in_channels, self.norm_pixel_loss, self.mlp_ratio = in_channels, norm_pixel_loss, mlp_ratio
        self.optimizer, self.schedule, self.betas, self.weight_decay = optimizer, schedule, betas, weight_decay
        self.lr, self.base_lr, self.min_lr, self.effective_bsz = None, base_lr, min_lr, effective_bsz
        self.warmup_epochs, self.max_epochs = warmup_epochs, max_epochs
        self.use_cls_token = use_cls_token
        self.language_dim = language_dim

        # Encoder/Decoder Parameters
        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.encoder_embed_dim, self.encoder_n_heads = encoder_embed_dim, encoder_n_heads
        self.decoder_embed_dim, self.decoder_n_heads = decoder_embed_dim, decoder_n_heads

        # General Parameters (for downstream adaptation)
        self.embed_dim, self.n_heads = self.encoder_embed_dim, self.encoder_n_heads

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        # MAE Encoder Parameters
        self.patch2embed = PatchEmbed(
            self.resolution, self.patch_size, self.encoder_embed_dim, in_channels=self.in_channels
        )
        self.encoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + (1 if self.use_cls_token else 0), self.encoder_embed_dim),
            requires_grad=False,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    self.encoder_embed_dim,
                    self.encoder_n_heads,
                    self.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                )
                for _ in range(self.encoder_depth)
            ]
        )
        self.encoder_norm = RMSNorm(self.encoder_embed_dim)

        # Projection from Language Embedding to Decoder
        self.lang2encoder = nn.Linear(self.language_dim, self.encoder_embed_dim)

        # Projection from Encoder to Decoder
        self.encoder2decoder = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)

        # MAE Decoder Parameters -- Remember the CLS Token (if specified)!
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.decoder_embed_dim))
        self.decoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + (1 if self.use_cls_token else 0), self.decoder_embed_dim),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    self.decoder_embed_dim,
                    self.decoder_n_heads,
                    self.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = RMSNorm(self.decoder_embed_dim)
        self.decoder_prediction = nn.Linear(self.decoder_embed_dim, (patch_size**2) * in_channels, bias=True)

        # VDual -- Add "Image" and "Language" Modifier Tokens...
        self.img_token = nn.Parameter(torch.zeros(1, 1, 1, self.encoder_embed_dim))
        self.lang_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        # VDual -- Learnable "ctx" position embeddings --> initialize via `randn` following original ViT & @lucidrains
        #   =>> Ref: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py#L99
        #       =>> Note that n_context = 2 (0th frame + Kth frame)
        self.ctx_enc_pe = nn.Parameter(torch.randn(1, 2, 1, self.encoder_embed_dim))
        self.ctx_dec_pe = nn.Parameter(torch.randn(1, 2, 1, self.decoder_embed_dim))

        # Initialize all Weights
        self.initialize_weights()

        # @AFTER INITIALIZATION -- Create Language Model & Language Reward MLP --> LM has requires_grad = False
        #   > For BERT models, our "embedding" is just going to be the last hidden state
        #   > Assumes inputs to forward pass are pre-tokenized!
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(language_model, cache_dir=hf_cache)
        self.lm = transformers.AutoModel.from_pretrained(language_model, cache_dir=hf_cache)
        self.lm.eval()

        # Shape Assertion -- make sure self.language_dim actually is the same as the LM dimension!
        assert self.lm.config.dim == self.language_dim, "Language model embedding dimension != self.language_dim!"

        # Freeze the LM
        for _, param in self.lm.named_parameters():
            param.requires_grad = False

    def initialize_weights(self) -> None:
        # Position Encoding -- Fixed 2D Sine-Cosine Embeddings
        enc_pe = get_2D_position_embeddings(
            self.encoder_embed_dim, int(self.patch2embed.num_patches**0.5), cls_token=self.use_cls_token
        )
        self.encoder_pe.data.copy_(torch.from_numpy(enc_pe).float().unsqueeze(0))
        dec_pe = get_2D_position_embeddings(
            self.decoder_embed_dim, int(self.patch2embed.num_patches**0.5), cls_token=self.use_cls_token
        )
        self.decoder_pe.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))

        # Initialize PatchEmbedding as a Linear...
        nn.init.xavier_uniform_(self.patch2embed.proj.weight.data.view([self.patch2embed.proj.weight.data.shape[0], -1]))

        # Initialize Mask Token, Img Token, Lang Token w/ Truncated Normal
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.img_token, std=0.02)
        nn.init.normal_(self.lang_token, std=0.02)
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

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

    def encode_language(self, lang: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """Encode language by feeding the *pre-tokenized text* through the frozen language model."""
        self.lm.eval()
        with torch.no_grad():
            transformer_embeddings = self.lm(lang, attention_mask=lang_mask).last_hidden_state
        return transformer_embeddings

    def mask(
        self, ctx_patches: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-context random masking by shuffling :: uses argsort random noise to identify masked patches."""
        bsz, ctx_len, n_patches, embed_dim = ctx_patches.shape
        if mask_ratio is not None:
            n_keep = int(n_patches * (1 - mask_ratio))
        else:
            n_keep = int(n_patches * (1 - self.mask_ratio))

        # Sample noise of n_patches size, argsort to get shuffled IDs, argsort again to get "unshuffle"
        #   > For clarity -- argsort is an invertible transformation (if argsort `restore`, recovers `shuffle`)
        #   > Note that shuffle_idxs is defined solely as a function of *n_patches* and **not** context! Same mask!
        shuffle_idxs = torch.argsort(torch.rand(bsz, n_patches, device=ctx_patches.device), dim=1)
        restore_idxs = torch.argsort(shuffle_idxs, dim=1)

        # Get "keep" (visible) patches --> make sure to get _same_ patches *across* context length!
        visible_patches = torch.gather(
            ctx_patches, dim=2, index=shuffle_idxs[:, None, :n_keep, None].repeat(1, ctx_len, 1, embed_dim)
        )

        # Generate the binary mask --> IMPORTANT :: `0` is keep, `1` is remove (following MAE convention)
        mask = torch.ones(bsz, n_patches, device=ctx_patches.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=restore_idxs)

        return visible_patches, mask, restore_idxs

    def get_representations(
        self, imgs: torch.Tensor, language: Optional[Union[List[str], Tuple[str]]] = None, mode: str = "multimodal"
    ) -> torch.Tensor:
        """
        Given either a singleton (dual-imgs, language) pair or batch of dual-imgs and language, extract representations
        subject to the specified mode in < multimodal | visual >.

        :param imgs: Processed batch of images :: [bsz, 2, 3, 224, 224]
        :param language: Input language as `List[str] | Tuple[str] | None`
        :param mode: Type of representations to extract -- `multimodal` (both vision+text), `visual` (visual only)

        :return: Extracted representations given (imgs, language) input as sequence.
        """
        assert (
            imgs.ndim == 5
            and imgs.shape[1] == 2
            and (language is None or isinstance(language, list) or isinstance(language, tuple))
        ), "Invalid input to `get_representations()`"
        assert mode in {"multimodal", "visual"}, f"Extraction mode `{mode}` not supported!"

        # Tokenize Language --> note max length is 20!
        if language is None:
            lang, lang_mask = [torch.zeros(imgs.shape[0], 20, dtype=int, device=self.lm.device) for _ in range(2)]
            lang[:, 0], lang_mask[:, 0] = self.tokenizer.cls_token_id, 1
        else:
            tokens = self.tokenizer(language, return_tensors="pt", max_length=20, padding="max_length", truncation=True)
            lang, lang_mask = tokens["input_ids"].to(self.lm.device), tokens["attention_mask"].to(self.lm.device)

            # Tile Language & Language Mask if mismatch with # images!
            if not len(lang) == len(imgs):
                lang = repeat(lang, "b seq -> (bsz b) seq", bsz=imgs.size(0))
                lang_mask = repeat(lang_mask, "b seq -> (bsz b) seq", bsz=imgs.size(0))

        # Extract desired representations...
        representations = self.encode(imgs, lang, lang_mask)
        return representations if mode == "multimodal" else representations[:, : -lang_mask.shape[-1]]

    def encode(self, imgs: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """Default representation extraction function, given a batch of dual-images and tokenized language."""
        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_language = self.lang2encoder(lang_embeddings)

        # Patchify, broadcast position embedding ctx_len (0 + K) dimension, unfold, add `ctx_enc_pe` embeddings!
        patches = self.patch2embed(rearrange(imgs, "bsz ctx channels res1 res2 -> (bsz ctx) channels res1 res2"))
        patches_pe = patches + (self.encoder_pe[:, 1:, :] if self.use_cls_token else self.encoder_pe)
        ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=2)
        ctx_patches_pe = ctx_patches + self.ctx_enc_pe[:, :2, ...]

        # Add "modality" embeddings to patches & language & flatten out context patches...
        img_ctx_embeddings, lang_embeddings = ctx_patches_pe + self.img_token, projected_language + self.lang_token
        img_embeddings = rearrange(img_ctx_embeddings, "bsz ctx seq embed -> bsz (ctx seq) embed")

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_token_pe = self.cls_token + self.encoder_pe[:, :1, :] + self.img_token[:, 0, :, :]
            cls_tokens = cls_token_pe.expand(imgs.shape[0], -1, -1)
            img_embeddings = torch.cat([cls_tokens, img_embeddings], dim=1)

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        patches_mask = torch.ones_like(img_embeddings[..., -1], dtype=lang_mask.dtype)
        multimodal_embeddings = torch.cat([img_embeddings, lang_embeddings], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([patches_mask, lang_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embeddings = block(multimodal_embeddings, multimodal_mask)
        multimodal_embeddings = self.encoder_norm(multimodal_embeddings)

        # Return the full sequence of multimodal embeddings (but ignore 0th frame) => the `~` denote what to remove!
        #   => [CLS] + ~[n_patches x 0th frame]~ + [n_patches x Kth frame] + [max_lang_len language]
        #   => ~[n_patches x 0th frame]~ + [n_patches x Kth frame] + [max_lang_len language]
        if self.use_cls_token:
            return torch.cat(
                [multimodal_embeddings[:, :1, :], multimodal_embeddings[:, 1 + self.patch2embed.num_patches :, :]], dim=1
            )
        else:
            return multimodal_embeddings[:, self.patch2embed.num_patches :]

    def forward_encoder(
        self, img_ctx: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_lang = self.lang2encoder(lang_embeddings)

        # Patchify, broadcast position embedding across ctx_len (0 + K) dimension, unfold, add `ctx_enc_pe` embeddings!
        patches = self.patch2embed(rearrange(img_ctx, "bsz ctx channels res1 res2 -> (bsz ctx) channels res1 res2"))
        patches_pe = patches + (self.encoder_pe if not self.use_cls_token else self.encoder_pe[:, 1:, :])
        ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=2)
        ctx_patches_pe = ctx_patches + self.ctx_enc_pe[:, :2, ...]

        # Create mask (and go ahead and mask out patches at the same time)
        visible_ctx_patches, mask, restore_idxs = self.mask(ctx_patches_pe, mask_ratio)

        # Add "modality" embeddings to patches & language & flatten out context patches...
        visible_ctx_patches, lang = visible_ctx_patches + self.img_token, projected_lang + self.lang_token
        visible_patches = rearrange(visible_ctx_patches, "bsz ctx seq embed -> bsz (ctx seq) embed")

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_token_pe = self.cls_token + self.encoder_pe[:, :1, :] + self.img_token[:, 0, :, :]
            cls_tokens = cls_token_pe.expand(img_ctx.shape[0], -1, -1)
            visible_patches = torch.cat([cls_tokens, visible_patches], dim=1)

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer...
        visible_mask = torch.ones_like(visible_patches[..., -1], dtype=lang_mask.dtype)
        multimodal_embedding = torch.cat([visible_patches, lang], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([visible_mask, lang_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embedding = block(multimodal_embedding, multimodal_mask)
        multimodal_embedding = self.encoder_norm(multimodal_embedding)

        # Split multimodal embedding, remove language, return the visible ctx (0th + Kth frame) patches (+ <CLS>)!
        visible_patches = multimodal_embedding[:, : -lang_mask.shape[-1], ...]
        return visible_patches, mask, restore_idxs

    def forward_decoder(self, visible_patches: torch.Tensor, restore_idxs: torch.Tensor) -> torch.Tensor:
        # Project patches into decoder embedding dimension (visible_ctx_patches :: [bsz, (CLS) + 2 * seq, enc_embed])
        projected_patches = self.encoder2decoder(visible_patches)
        visible_per_frame = (projected_patches.shape[1] - (1 if self.use_cls_token else 0)) // 2

        # Add Mask Tokens to Sequence and Unshuffle
        mask_tokens = self.mask_token.repeat(projected_patches.shape[0], 2, restore_idxs.shape[1] - visible_per_frame, 1)

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            # Remove CLS Token as part of "unshuffling"
            projected_ctx_patches = rearrange(
                projected_patches[:, 1:, :], "bsz (ctx seq) embed -> bsz ctx seq embed", ctx=2
            )
            no_cls_concatenated_ctx_patches = torch.cat([projected_ctx_patches, mask_tokens], dim=2)
            unshuffled_ctx_patches = torch.gather(
                no_cls_concatenated_ctx_patches,
                dim=2,
                index=restore_idxs[:, None, ..., None].repeat(1, 2, 1, self.decoder_embed_dim),
            )
        else:
            projected_ctx_patches = rearrange(projected_patches, "bsz (ctx seq) embed -> bsz ctx seq embed", ctx=2)
            concatenated_ctx_patches = torch.cat([projected_ctx_patches, mask_tokens], dim=2)
            unshuffled_ctx_patches = torch.gather(
                concatenated_ctx_patches,
                dim=2,
                index=restore_idxs[:, None, ..., None].repeat(1, 2, 1, self.decoder_embed_dim),
            )

        # Add position embeddings, `ctx_dec_pe` embeddings, and flatten patches for Transformer...
        decoder_ctx_patches_pe = unshuffled_ctx_patches + (
            self.decoder_pe[None, ...] if not self.use_cls_token else self.decoder_pe[None, :, 1:, :]
        )
        decoder_ctx_patches = decoder_ctx_patches_pe + self.ctx_dec_pe[:, :2, ...]
        decoder_patches = rearrange(decoder_ctx_patches, "bsz ctx seq embed -> bsz (ctx seq) embed")

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            # Add back <CLS> Token from `projected_patches[:, :1, :]`
            cls_embedding = projected_patches[:, :1, :] + self.decoder_pe[:, :1, :]
            decoder_patches = torch.cat([cls_embedding, decoder_patches], dim=1)

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            decoder_patches = block(decoder_patches)
        decoder_patches = self.decoder_norm(decoder_patches)

        # Run final projection & return "unflattened" patches --> note <CLS> token handling!
        decoder_prediction = self.decoder_prediction(decoder_patches)
        reconstructions = decoder_prediction if not self.use_cls_token else decoder_prediction[:, 1:, :]
        return rearrange(reconstructions, "bsz (ctx seq) embed -> bsz ctx seq embed", ctx=2)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of (0th + Kth frame) images to their patched equivalents by naive reshaping."""
        return rearrange(
            imgs,
            "bsz ctx c (height patch_h) (width patch_w) -> bsz ctx (height width) (patch_h patch_w c)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )

    def compute_loss(
        self, imgs: torch.Tensor, ctx_reconstructions: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_pixel_loss, "`norm_pixel_loss` should always be true... false only for visualizations!"
        targets = self.patchify(imgs)

        # Normalize targets...
        mu, var = targets.mean(dim=-1, keepdim=True), targets.var(dim=-1, unbiased=True, keepdim=True)
        targets = (targets - mu) / ((var + 1e-6) ** 0.5)

        # Split targets into 0 and K --> do the same for ctx_reconstructions
        zero_target, k_target = targets[:, 0, ...], targets[:, 1, ...]
        zero_reconstruction, k_reconstruction = ctx_reconstructions[:, 0, ...], ctx_reconstructions[:, 1, ...]

        # Compute mean losses per patch first...
        zero_mse, k_mse = (zero_reconstruction - zero_target) ** 2, (k_reconstruction - k_target) ** 2
        zero_avg_loss_per_patch, k_avg_loss_per_patch = zero_mse.mean(dim=-1), k_mse.mean(dim=-1)

        # Compute mean loss only on *removed* patches and return...
        return (zero_avg_loss_per_patch * mask).sum() / mask.sum(), (k_avg_loss_per_patch * mask).sum() / mask.sum()

    def forward(
        self, imgs: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run a forward pass through the model, computing the language-conditioned MAE reconstruction loss on the
        0th + Kth frame temporal context, given language prefix.

        :param imgs: A [bsz, 2, in_channels, resolution, resolution] tensor of (0th frame, Kth frame) sequences.
        :param lang: A [bsz, seq_len] tensor of language context to condition on.
        :param lang_mask: A [bsz, seq_len] binary mask tensor to indicate padding locations in the lang tensor.
        :param mask_ratio: Optional masking ratio to use instead of the default.

        :return Tuple of losses and intermediates, as follows:
            > (combined loss, [reconstruction loss per frame in {0, K}])
        """
        visible_ctx_patches, mask, restore_idxs = self.forward_encoder(imgs, lang, lang_mask, mask_ratio)
        ctx_reconstructions = self.forward_decoder(visible_ctx_patches, restore_idxs)
        zero_loss, k_loss = self.compute_loss(imgs, ctx_reconstructions, mask)

        # Return average reconstruction loss, individual losses...
        loss = (zero_loss + k_loss) / 2
        return loss, [zero_loss, k_loss]

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
