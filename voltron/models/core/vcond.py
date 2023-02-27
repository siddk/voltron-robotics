"""
vcond.py

PyTorch Module defining the Voltron `V-Cond` variant (single-frame with language-conditioning). In general, follows the
MAE recipe, with the architectural modifications described in the paper:
    - RMSNorm, for stability/performance ("Do Transformer Modifications Transfer...")
    - SwishGLU activations in the Attention Block Feed-Forward MLP (gated linear units) as used in PaLM
    - LayerScale with a default value of 0.1 (from Mistral/CaIT)

References:
    - https://github.com/facebookresearch/mae
    - https://github.com/lucidrains/x-transformers
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


class VCond(nn.Module):
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
        effective_bsz: float,
        betas: Tuple[float, float],
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        mask_ratio: float = 0.75,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        norm_pixel_loss: bool = True,
    ) -> None:
        """
        Initialize a VCond model with the requisite architecture parameters.

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
        :param mlp_ratio: Ratio for embedding size to Position-wise Feed-Forward MLP (gets shrunk back down).
        :param in_channels: Default number of channels in the base image -- almost always 3.
        :param norm_pixel_loss: Normalize decoder pixel targets for reconstruction (better perf, not interpretable).
        """
        super().__init__()
        self.resolution, self.patch_size, self.mask_ratio = resolution, patch_size, mask_ratio
        self.in_channels, self.norm_pixel_loss, self.mlp_ratio = in_channels, norm_pixel_loss, mlp_ratio
        self.optimizer, self.schedule, self.betas, self.weight_decay = optimizer, schedule, betas, weight_decay
        self.lr, self.base_lr, self.min_lr, self.effective_bsz = None, base_lr, min_lr, effective_bsz
        self.warmup_epochs, self.max_epochs = warmup_epochs, max_epochs
        self.language_dim = language_dim

        # Encoder/Decoder Parameters
        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.encoder_embed_dim, self.encoder_n_heads = encoder_embed_dim, encoder_n_heads
        self.decoder_embed_dim, self.decoder_n_heads = decoder_embed_dim, decoder_n_heads

        # General Parameters (for downstream adaptation)
        self.embed_dim, self.n_heads = self.encoder_embed_dim, self.encoder_n_heads

        # MAE Encoder Parameters --> No CLS Token!
        self.patch2embed = PatchEmbed(
            self.resolution, self.patch_size, self.encoder_embed_dim, in_channels=self.in_channels
        )
        self.encoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches, self.encoder_embed_dim), requires_grad=False
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

        # MAE Decoder Parameters -- Remember the MASK Token!
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches, self.decoder_embed_dim), requires_grad=False
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

        # VCond -- Add "Image" and "Language" Modifier Tokens...
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.lang_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

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
        enc_pe = get_2D_position_embeddings(self.encoder_embed_dim, int(self.patch2embed.num_patches**0.5))
        self.encoder_pe.data.copy_(torch.from_numpy(enc_pe).float().unsqueeze(0))
        dec_pe = get_2D_position_embeddings(self.decoder_embed_dim, int(self.patch2embed.num_patches**0.5))
        self.decoder_pe.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))

        # Initialize PatchEmbedding as a Linear...
        nn.init.xavier_uniform_(self.patch2embed.proj.weight.data.view([self.patch2embed.proj.weight.data.shape[0], -1]))

        # Initialize Mask Token, Img Token, Lang Token w/ Truncated Normal
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.img_token, std=0.02)
        nn.init.normal_(self.lang_token, std=0.02)

        # Default Transformer initializer on everything else...
        self.apply(self.transformer_initializer)

    @staticmethod
    def transformer_initializer(m: nn.Module) -> None:
        # Use `xavier_uniform` following Jax ViT
        if isinstance(m, nn.Linear):
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
        self, patches: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-sample random masking by shuffling :: uses argsort random noise to identify masked patches."""
        bsz, n_patches, embed_dim = patches.shape
        if mask_ratio is not None:
            n_keep = int(n_patches * (1 - mask_ratio))
        else:
            n_keep = int(n_patches * (1 - self.mask_ratio))

        # Sample noise of n_patches size, argsort to get shuffled IDs, argsort again to get "unshuffle"
        #   > For clarity -- argsort is an invertible transformation (if argsort `restore`, recovers `shuffle`)
        shuffle_idxs = torch.argsort(torch.rand(bsz, n_patches, device=patches.device), dim=1)
        restore_idxs = torch.argsort(shuffle_idxs, dim=1)

        # Get "keep" (visible) patches
        visible_patches = torch.gather(patches, dim=1, index=shuffle_idxs[:, :n_keep, None].repeat(1, 1, embed_dim))

        # Generate the binary mask --> IMPORTANT :: `0` is keep, `1` is remove (following MAE convention)
        mask = torch.ones(bsz, n_patches, device=patches.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=restore_idxs)

        return visible_patches, mask, restore_idxs

    def get_representations(
        self, img: torch.Tensor, language: Optional[Union[List[str], Tuple[str]]] = None, mode: str = "multimodal"
    ) -> torch.Tensor:
        """
        Given either a singleton (img, language) pair or a batch of images and language, extract representations
        subject to the specified mode in < multimodal | visual >.

        :param img: Processed batch of images :: [bsz, 3, 224, 224]
        :param language: Input language as `List[str] | Tuple[str] | None`
        :param mode: Type of representations to extract -- `multimodal` (both vision+text), `visual` (visual only)

        :return: Extracted representations given (img, language) input as sequence.
        """
        assert img.ndim == 4 and (
            language is None or isinstance(language, list) or isinstance(language, tuple)
        ), "Invalid input to `get_representations()`"
        assert mode in {"multimodal", "visual"}, f"Extraction mode `{mode}` not supported!"

        # Tokenize Language --> note max length is 20!
        if language is None:
            lang, lang_mask = [torch.zeros(img.shape[0], 20, dtype=int, device=self.lm.device) for _ in range(2)]
            lang[:, 0], lang_mask[:, 0] = self.tokenizer.cls_token_id, 1
        else:
            tokens = self.tokenizer(language, return_tensors="pt", max_length=20, padding="max_length", truncation=True)
            lang, lang_mask = tokens["input_ids"].to(self.lm.device), tokens["attention_mask"].to(self.lm.device)

            # Tile Language & Language Mask if mismatch with # images!
            if not len(lang) == len(img):
                lang = repeat(lang, "b seq -> (bsz b) seq", bsz=img.size(0))
                lang_mask = repeat(lang_mask, "b seq -> (bsz b) seq", bsz=img.size(0))

        # Extract desired representations...
        representations = self.encode(img, lang, lang_mask)
        return representations if mode == "multimodal" else representations[:, : -lang_mask.shape[-1]]

    def encode(self, img: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """Default representation extraction function, given a batch of images and tokenized language."""
        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_language = self.lang2encoder(lang_embeddings)

        # Patchify + Position Embedding
        patches = self.patch2embed(img)
        patches_pe = patches + self.encoder_pe

        # Add "modality" embeddings to patches & language
        img_embeddings, lang_embeddings = patches_pe + self.img_token, projected_language + self.lang_token

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        patches_mask = torch.ones_like(img_embeddings[..., -1], dtype=lang_mask.dtype)
        multimodal_embeddings = torch.cat([img_embeddings, lang_embeddings], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([patches_mask, lang_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embeddings = block(multimodal_embeddings, multimodal_mask)
        multimodal_embeddings = self.encoder_norm(multimodal_embeddings)

        # Return the full sequence of multimodal embeddings...
        return multimodal_embeddings

    def forward_encoder(
        self, img: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Patchify + Position Embedding
        patches = self.patch2embed(img)
        patches_pe = patches + self.encoder_pe

        # Create mask (and go ahead and mask out patches at the same time)
        visible_patches, mask, restore_idxs = self.mask(patches_pe, mask_ratio)

        # Add "modality" embeddings to patches & language
        visible_patches, lang = visible_patches + self.img_token, lang + self.lang_token

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        visible_mask = torch.ones_like(visible_patches[..., -1], dtype=lang_mask.dtype)
        multimodal_embedding = torch.cat([visible_patches, lang], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([visible_mask, lang_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embedding = block(multimodal_embedding, multimodal_mask)
        multimodal_embedding = self.encoder_norm(multimodal_embedding)

        # Split multimodal embedding, remove language and return only the visible patches!
        visible_patches = multimodal_embedding[:, : -lang_mask.shape[-1], ...]
        return visible_patches, mask, restore_idxs

    def forward_decoder(self, visible_patches: torch.Tensor, restore_idxs: torch.Tensor) -> torch.Tensor:
        # Project patches into decoder embedding dimension
        projected_patches = self.encoder2decoder(visible_patches)

        # Add Mask Tokens to Sequence & Unshuffle
        mask_tokens = self.mask_token.repeat(
            projected_patches.shape[0], restore_idxs.shape[1] - visible_patches.shape[1], 1
        )
        concatenated_patches = torch.cat([projected_patches, mask_tokens], dim=1)
        unshuffled_patches = torch.gather(
            concatenated_patches, dim=1, index=restore_idxs[..., None].repeat(1, 1, self.decoder_embed_dim)
        )

        # Add Position Embeddings
        decoder_patches = unshuffled_patches + self.decoder_pe

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            decoder_patches = block(decoder_patches)
        decoder_patches = self.decoder_norm(decoder_patches)

        # Run final projection & return
        return self.decoder_prediction(decoder_patches)

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
        self, img: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # First, get precomputed language embeddings & project to encoder_embed_dim; language only used in encoder!
        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_language = self.lang2encoder(lang_embeddings)

        # MAE --> does image masking in the body of the model...
        visible_patches, mask, restore_idxs = self.forward_encoder(img, projected_language, lang_mask, mask_ratio)
        reconstructions = self.forward_decoder(visible_patches, restore_idxs)
        loss = self.compute_loss(img, reconstructions, mask)
        return loss, reconstructions, mask

    def configure_optimizer(self) -> Tuple[torch.optim.Optimizer, Callable[[int, float], float]]:
        # Short-circuit on valid optimizers
        if self.optimizer not in ["adamw"]:
            raise NotImplementedError(f"Optimizer `{self.optimizer}` not supported - try [`adamw`] instead!")

        # Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed...
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
