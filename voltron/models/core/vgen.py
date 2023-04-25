"""
vgen.py

PyTorch Module defining the Voltron `V-Gen` variant (dual-frame with language-conditioning AND language-generation).
This model adds the ability to *both* condition on language context or (XOR) generate language given masked frame
context (with a hyperparameter (`alpha` in the paper) controlling the `gen_ratio` -- the ratio of examples for which to
generate language).

The objective this model seeks to optimize is the sum of the MAE reconstruction error (when conditioning on language)
and the log-likelihood of predicting the next token given prior tokens and the entire learned image representation.

Follows same dual-frame encoding structure as VDual, and architectural modifications from VCond.

References:
    - https://github.com/lucidrains/x-transformers
"""
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from einops import rearrange, repeat

from voltron.models.util.optimization import get_lr_update
from voltron.models.util.transformer import Block, PatchEmbed, RMSNorm, get_2D_position_embeddings

# Suppress Transformers Logging
transformers.logging.set_verbosity_error()


class VGen(nn.Module):
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
        max_lang_len: int,
        vocab_size: int,
        mae_weight: float,
        lm_weight: float,
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
        use_cls_token: bool = False,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize a VGen model with the requisite architecture parameters.

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
        :param max_lang_len: Maximum length of input sequence (in tokens).
        :param vocab_size: Vocabulary size for final cross-entropy loss over token prediction.
        :param mae_weight: Weighting term for the MAE loss -- usually 1.0 (borrowed from M3AE paper as *rough* guide)
        :param lm_weight: Weighting term for the LM loss -- usually 0.5 (borrowed from the M3AE paper as *rough* guide)
        :param optimizer: String denoting which optimizer to use (for MAEs, usually `adamw`)
        :param schedule: Learning rate schedule to use; for Transformers a linear warmup + decay is recommended!
        :param base_lr: Base learning rate, to be scaled via a linear scaling rule (from scaling laws).
        :param min_lr: Minimum learning rate to decay to over the course of learning (usually 0.0)
        :param effective_bsz: Global batch size for update, dictates the scaling of the base_lr.
        :param betas: Adam optimizer betas (only applicable for `adam` and `adamw`. Prevents early loss spiking.
        :param weight_decay: Weight decay for global weight regularization (only applied to non-bias, non-LN layers).
        :param warmup_epochs: Number of epochs to warmup learning rate for linear warmup schedule.
        :param max_epochs: Total number of training epochs to be run.
        :param mask_ratio: Ratio for number of patches AND tokens to mask out for M3AE -- should be fairly high!
        :param mlp_ratio: Ratio for embedding size to Position-wise FeedForward MLP (gets shrunk back down).
        :param in_channels: Default number of channels in the base image -- almost always 3.
        :param norm_pixel_loss: Normalize decoder pixel targets for reconstruction (better perf, not interpretable).
        :param use_cls_token: Add <CLS> token for continued pretraining (NOTE: not used in MAE pretraining/finetuning!)
        :param eps: Epsilon for preventing divide by zero.
        """
        super().__init__()
        self.resolution, self.patch_size, self.mask_ratio, self.eps = resolution, patch_size, mask_ratio, eps
        self.in_channels, self.norm_pixel_loss, self.mlp_ratio = in_channels, norm_pixel_loss, mlp_ratio
        self.optimizer, self.schedule, self.betas, self.weight_decay = optimizer, schedule, betas, weight_decay
        self.lr, self.base_lr, self.min_lr, self.effective_bsz = None, base_lr, min_lr, effective_bsz
        self.mae_weight, self.lm_weight, self.language_dim = mae_weight, lm_weight, language_dim
        self.max_lang_len, self.vocab_size = max_lang_len, vocab_size
        self.use_cls_token = use_cls_token
        self.warmup_epochs, self.max_epochs = warmup_epochs, max_epochs

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
        self.lang2decoder = nn.Linear(self.language_dim, self.decoder_embed_dim)

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
        self.decoder_patch_prediction = nn.Linear(self.decoder_embed_dim, (patch_size**2) * in_channels, bias=True)
        self.decoder_lang_prediction = nn.Linear(self.decoder_embed_dim, self.vocab_size, bias=True)

        # VGen -- Add "Image" and "Language" Modifier Tokens for Encoder & Decoder...
        self.img_enc_token = nn.Parameter(torch.zeros(1, 1, 1, self.encoder_embed_dim))
        self.lang_enc_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        # VGen -- Learnable "ctx" position embeddings --> initialize via `randn` following original ViT & @lucidrains
        #   =>> Ref: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py#L99
        #       =>> Note that n_context = 2 (0th frame + Kth frame)
        self.ctx_enc_pe = nn.Parameter(torch.randn(1, 2, 1, self.encoder_embed_dim))
        self.ctx_dec_pe = nn.Parameter(torch.randn(1, 2, 1, self.decoder_embed_dim))

        # Register Prefix Mask --> Lower Triangular ==> set prefix to 1
        n_patches, total_seq = 2 * self.patch2embed.num_patches, (2 * self.patch2embed.num_patches) + self.max_lang_len
        prefix_mask = torch.tril(torch.ones((total_seq, total_seq), dtype=torch.uint8))
        prefix_mask[:n_patches, :n_patches] = 1

        # Register this once... we'll multiply by padding masks prior to feeding to Transformer
        self.register_buffer("prefix_mask", prefix_mask.view(1, 1, total_seq, total_seq))

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
        nn.init.normal_(self.img_enc_token, std=0.02)
        nn.init.normal_(self.lang_enc_token, std=0.02)
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

    def embed_language(self, lang: torch.Tensor) -> torch.Tensor:
        """Only feed language through the pretrained *embedding* matrix (no bidirectional cheating)."""
        self.lm.eval()
        with torch.no_grad():
            # Note :: These have position_encodings included... no need for separate `decoder_lang_pe`
            token_embeddings = self.lm.embeddings(lang)
        return token_embeddings

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

        # Generate the binary mask --> IMPORTANT :: `0` is keep, `1` is remove (following FAIR MAE convention)
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

        # Patchify, broadcast position embedding across ctx_len (0 + K) dimension, unfold, add `ctx_enc_pe` embeddings!
        patches = self.patch2embed(rearrange(imgs, "bsz ctx channels res1 res2 -> (bsz ctx) channels res1 res2"))
        patches_pe = patches + (self.encoder_pe[:, 1:, :] if self.use_cls_token else self.encoder_pe)
        ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=2)
        ctx_patches_pe = ctx_patches + self.ctx_enc_pe[:, :2, ...]

        # Add "modality" embeddings to patches & language & flatten out context patches...
        img_ctx_embeddings, lang_embeddings = (
            ctx_patches_pe + self.img_enc_token,
            projected_language + self.lang_enc_token,
        )
        img_embeddings = rearrange(img_ctx_embeddings, "bsz ctx seq embed -> bsz (ctx seq) embed")

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_token_pe = self.cls_token + self.encoder_pe[:, :1, :] + self.img_enc_token[:, 0, :, :]
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

    def score(self, imgs: torch.Tensor, langs: torch.Tensor, lang_masks: torch.Tensor) -> torch.Tensor:
        """
        Given an example 0-K pair and a set of k language instructions, output scores under the generative language
        model for each instruction.

        :param imgs: 0-K pairs --> [1, 2, 3, 224, 224]
        :param langs: Tokenized language input --> [1, k, seq]
        :param lang_masks: Language padding masks --> [1, k, seq]

        :return: [1, k] Tensor of LM probabilities given imgs.
        """
        # Blank out the "encoder" language --> just [<CLS> = 101, 0 ...]
        blank_lang = torch.zeros(1, self.max_lang_len, dtype=torch.int64, device=imgs.device)
        blank_lang_mask = torch.zeros(1, self.max_lang_len, dtype=torch.int64, device=imgs.device)
        blank_lang[0][0], blank_lang_mask[0][0] = 101, 1

        # === Encoder Forward ===
        lang_embeddings = self.encode_language(blank_lang, blank_lang_mask)
        projected_language = self.lang2encoder(lang_embeddings)

        # Patchify, broadcast position embedding across ctx_len (0 + K) dimension, unfold, add `ctx_enc_pe` embeddings!
        patches = self.patch2embed(rearrange(imgs, "bsz ctx channels res1 res2 -> (bsz ctx) channels res1 res2"))
        patches_pe = patches + (self.encoder_pe[:, 1:, :] if self.use_cls_token else self.encoder_pe)
        ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=2)
        ctx_patches_pe = ctx_patches + self.ctx_enc_pe[:, :2, ...]

        # Add "modality" embeddings to patches & language & flatten out context patches...
        img_ctx_embeddings, lang_embeddings = (
            ctx_patches_pe + self.img_enc_token,
            projected_language + self.lang_enc_token,
        )
        img_embeddings = rearrange(img_ctx_embeddings, "bsz ctx seq embed -> bsz (ctx seq) embed")

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_token_pe = self.cls_token + self.encoder_pe[:, :1, :] + self.img_enc_token[:, 0, :, :]
            cls_tokens = cls_token_pe.expand(imgs.shape[0], -1, -1)
            img_embeddings = torch.cat([cls_tokens, img_embeddings], dim=1)

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer
        patches_mask = torch.ones_like(img_embeddings[..., -1], dtype=blank_lang_mask.dtype)
        multimodal_embeddings = torch.cat([img_embeddings, lang_embeddings], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([patches_mask, blank_lang_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embeddings = block(multimodal_embeddings, multimodal_mask)
        multimodal_embeddings = self.encoder_norm(multimodal_embeddings)

        # Split multimodal embedding, remove language, and return only the (CLS +) 0th + Kth frame patches
        enc_patches = multimodal_embeddings[:, : -blank_lang_mask.shape[-1], ...]

        # === Encoder =>> Decoder Hand-Off ===
        enc_patches = repeat(enc_patches, "b cseq embed -> (bsz b) cseq embed", bsz=langs.size(0))
        lang_gen_embeddings = self.embed_language(langs)

        # === Decoder Forward ===
        projected_patches = self.encoder2decoder(enc_patches)
        projected_lang_gen = self.lang2decoder(lang_gen_embeddings)

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            projected_ctx_patches = rearrange(
                projected_patches[:, 1:, :], "bsz (ctx seq) embed -> bsz ctx seq embed", ctx=2
            )
        else:
            projected_ctx_patches = rearrange(projected_patches, "bsz (ctx seq) embed -> bsz ctx seq embed", ctx=2)

        # Add position embeddings, `ctx_dec_pe` embeddings, and flatten patches for Transformer...
        decoder_ctx_patches_pe = projected_ctx_patches + (
            self.decoder_pe[None, ...] if not self.use_cls_token else self.decoder_pe[None, :, 1:, :]
        )
        decoder_ctx_patches = decoder_ctx_patches_pe + self.ctx_dec_pe[:, :2, ...]
        decoder_patches = rearrange(decoder_ctx_patches, "bsz ctx seq embed -> bsz (ctx seq) embed")

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            # Add back <CLS> Token from `projected_patches[:, :1, :]`
            cls_embedding = projected_patches[:, :1, :] + self.decoder_pe[:, :1, :]
            decoder_patches = torch.cat([cls_embedding, decoder_patches], dim=1)

        # Add language -> create "mask" by multiply padding by self.prefix_mask
        decoder_patches_mask = torch.ones_like(decoder_patches[..., -1], dtype=lang_masks.dtype)
        multimodal_embedding = torch.cat([decoder_patches, projected_lang_gen], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([decoder_patches_mask, lang_masks], dim=1)  # Merge on sequence length...

        # Compute prefix_padded_mask
        prefix_padded_mask = rearrange(multimodal_mask, "bsz seq -> bsz 1 seq 1") * self.prefix_mask

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            multimodal_embedding = block(multimodal_embedding, prefix_padded_mask)
        multimodal_embedding = self.decoder_norm(multimodal_embedding)

        # Split multimodal embedding into *just* the language + project!
        lang = multimodal_embedding[:, -lang_masks.shape[-1] :, ...]
        generations = self.decoder_lang_prediction(lang)

        # Compute cross-entropy loss (multiply by -1 for "final scoring") --> log-likelihood!
        bsz, seq = langs.shape
        lang_logits = rearrange(generations[:, :-1, ...], "bsz seq vocab -> (bsz seq) vocab")
        lang_targets = rearrange(langs[:, 1:], "bsz seq -> (bsz seq)")
        lang_loss_mask = lang_masks[:, :-1]  # Defined where valid...
        ce_loss = F.cross_entropy(lang_logits, lang_targets, reduction="none")
        per_token_loss = rearrange(ce_loss, "(bsz seq) -> bsz seq", bsz=bsz, seq=seq - 1)  # -1 because causal mask...

        # Compute loss only on *non-padded* and *non-ignored* tokens...
        lang_example_loss = (per_token_loss * lang_loss_mask).sum(dim=-1) / lang_loss_mask.sum(dim=-1)
        return -1 * lang_example_loss.detach()

    def forward_encoder(
        self,
        img_ctx: torch.Tensor,
        lang_con: torch.Tensor,
        lang_con_mask: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lang_embeddings = self.encode_language(lang_con, lang_con_mask)
        projected_lang = self.lang2encoder(lang_embeddings)

        # Reshape image context to apply masking *identically*

        # Patchify, broadcast position embedding across ctx_len (0 + K) dimension, unfold, add `ctx_enc_pe` embeddings!
        patches = self.patch2embed(rearrange(img_ctx, "bsz ctx channels res1 res2 -> (bsz ctx) channels res1 res2"))
        patches_pe = patches + (self.encoder_pe if not self.use_cls_token else self.encoder_pe[:, 1:, :])
        ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=2)
        ctx_patches_pe = ctx_patches + self.ctx_enc_pe[:, :2, ...]

        # Create mask (and go ahead and mask out patches at the same time)
        visible_ctx_patches, mask, restore_idxs = self.mask(ctx_patches_pe, mask_ratio)

        # Add "modality" embeddings to patches & language & flatten out context patches...
        visible_ctx_patches, lang = visible_ctx_patches + self.img_enc_token, projected_lang + self.lang_enc_token
        visible_patches = rearrange(visible_ctx_patches, "bsz ctx seq embed -> bsz (ctx seq) embed")

        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            cls_token_pe = self.cls_token + self.encoder_pe[:, :1, :] + self.img_enc_token[:, 0, :, :]
            cls_tokens = cls_token_pe.expand(img_ctx.shape[0], -1, -1)
            visible_patches = torch.cat([cls_tokens, visible_patches], dim=1)

        # Create "dummy" visible mask, concatenate image patches & language, feed to Transformer...
        visible_mask = torch.ones_like(visible_patches[..., -1], dtype=lang_con_mask.dtype)
        multimodal_embedding = torch.cat([visible_patches, lang], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([visible_mask, lang_con_mask], dim=1)  # Merge on sequence length...

        # Apply Transformer Blocks...
        for block in self.encoder_blocks:
            multimodal_embedding = block(multimodal_embedding, multimodal_mask)
        multimodal_embedding = self.encoder_norm(multimodal_embedding)

        # Split multimodal embedding, remove language, return the visible ctx (0th + Kth frame) patches (+ <CLS>)!
        visible_patches = multimodal_embedding[:, : -lang_con_mask.shape[-1], ...]
        return visible_patches, mask, restore_idxs

    def forward_decoder(
        self,
        visible_patches: torch.Tensor,
        restore_idxs: torch.Tensor,
        lang_gen: torch.Tensor,
        lang_gen_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project patches & lang_gen into decoder dimension (visible_patches :: [bsz, (CLS) + 2 * seq, enc_embed])
        projected_patches = self.encoder2decoder(visible_patches)
        projected_lang_gen = self.lang2decoder(lang_gen)
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

        # Add language -> create "mask" by multiply padding by self.prefix_mask
        decoder_patches_mask = torch.ones_like(decoder_patches[..., -1], dtype=lang_gen_mask.dtype)
        multimodal_embedding = torch.cat([decoder_patches, projected_lang_gen], dim=1)  # Merge on sequence length...
        multimodal_mask = torch.cat([decoder_patches_mask, lang_gen_mask], dim=1)  # Merge on sequence length...

        # Compute prefix_padded_mask
        prefix_padded_mask = rearrange(multimodal_mask, "bsz seq -> bsz 1 seq 1") * self.prefix_mask

        # Apply Transformer Blocks...
        for block in self.decoder_blocks:
            multimodal_embedding = block(multimodal_embedding, prefix_padded_mask)
        multimodal_embedding = self.decoder_norm(multimodal_embedding)

        # Split multimodal embedding into patches and language...
        patches_ctx = multimodal_embedding[:, : -lang_gen_mask.shape[-1], ...]
        patches = rearrange(
            patches_ctx if not self.use_cls_token else patches_ctx[:, 1:, :],
            "bsz (ctx seq) embed -> bsz ctx seq embed",
            ctx=2,
        )
        lang = multimodal_embedding[:, -lang_gen_mask.shape[-1] :, ...]

        # Project each up to the output space...
        reconstructions = self.decoder_patch_prediction(patches)
        generations = self.decoder_lang_prediction(lang)

        return reconstructions, generations

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of (0th + Kth frame) images to their patched equivalents by naive reshaping."""
        return rearrange(
            imgs,
            "bsz ctx c (height patch_h) (width patch_w) -> bsz ctx (height width) (patch_h patch_w c)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )

    def compute_loss(
        self,
        imgs: torch.Tensor,
        ctx_reconstructions: torch.Tensor,
        mask: torch.Tensor,
        lang: torch.Tensor,
        generated_language: torch.Tensor,
        lang_gen_mask: torch.Tensor,
        lang_gen_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction Loss...
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

        # Compute reconstruction losses...
        zero_loss = (zero_avg_loss_per_patch * mask).sum() / mask.sum()
        k_loss = (k_avg_loss_per_patch * mask).sum() / mask.sum()
        reconstruction_loss = (zero_loss + k_loss) / 2

        # Language Loss...
        bsz, seq = lang.shape
        lang_logits = rearrange(generated_language[:, :-1, ...], "bsz seq vocab -> (bsz seq) vocab")
        lang_targets = rearrange(lang[:, 1:], "bsz seq -> (bsz seq)")
        lang_loss_mask = lang_gen_mask[:, :-1]  # Defined where valid...
        ce_loss = F.cross_entropy(lang_logits, lang_targets, reduction="none")
        per_token_loss = rearrange(ce_loss, "(bsz seq) -> bsz seq", bsz=bsz, seq=seq - 1)  # -1 because causal mask...

        # Compute loss only on *non-padded* and *non-ignored* tokens...
        lang_example_loss = (per_token_loss * lang_loss_mask).sum(dim=-1) / lang_loss_mask.sum(dim=-1)
        lang_loss = (lang_example_loss * lang_gen_weight).sum() / (self.eps + lang_gen_weight.sum())  # Divide by 0...

        # TODO (Remove) -- NaN Check...
        if reconstruction_loss.isnan().any() or lang_loss.isnan().any():
            # fmt: off
            print(
                f"Found Nan -- "
                f"ctx_reconstructions: {ctx_reconstructions.isnan().any()} -- "
                f"generated_language: {generated_language.isnan().any()} -- "
                f"zero_avg_loss_per_patch: {zero_avg_loss_per_patch.isnan().any()} -- "
                f"k_avg_loss_per_patch: {k_avg_loss_per_patch.isnan().any()} -- "
                f"zero_loss: {zero_loss.isnan().any()} -- "
                f"k_loss: {k_loss.isnan().any()} -- "
                f"reconstruction_loss: {reconstruction_loss.isnan().any()} -- "
                f"ce_loss: {ce_loss.isnan().any()} -- "
                f"per_token_loss: {per_token_loss.isnan().any()} -- "
                f"lang_example_loss: {lang_example_loss.isnan().any()} -- "
                f"lang_loss: {lang_loss.isnan().any()}"
            )
            exit(1)
            # fmt: on

        # Compute weighted loss...
        loss = self.mae_weight * reconstruction_loss + self.lm_weight * lang_loss
        return loss, reconstruction_loss, lang_loss, zero_loss, k_loss

    def forward(
        self,
        imgs: torch.Tensor,
        lang_con: torch.Tensor,
        lang_con_mask: torch.Tensor,
        lang_gen: torch.Tensor,
        lang_gen_mask: torch.Tensor,
        lang_gen_weight: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Run a forward pass through the model, computing the MAE reconstruction loss (language-conditioned if applicable)
        on the 0th + Kth frame temporal context, as well as the generated language given masked context (if applicable).

        :param imgs: A [bsz, 2, in_channels, resolution, resolution] tensor of (0th frame, Kth frame) sequences.
        :param lang_con: A [bsz, seq_len] tensor of language context to condition on.
        :param lang_con_mask: A [bsz, seq_len] binary mask tensor to indicate padding/null in `lang_condition`.
        :param lang_gen: A [bsz, seq_len] tensor of language to generate.
        :param lang_gen_mask: A [bsz, seq_len] binary mask tensor to indicate padding/null in `lang_gen`.
        :param lang_gen_weight: A [bsz] tensor of per-example weights to indicate when to 0 `lm` loss.
        :param mask_ratio: Optional masking ratio to use instead of the default.

        :return: Tuple of losses and intermediates, as follows:
            > (combined loss, reconstruction loss, lm loss, [reconstruction loss per frame in {0, K}])
        """
        visible_ctx_patches, mask, restore_idxs = self.forward_encoder(imgs, lang_con, lang_con_mask, mask_ratio)

        # Get token embeddings -- *NOT CONTEXTUAL* -- for the lang_gen tokens...
        lang_gen_embeddings = self.embed_language(lang_gen)

        # Run patches, and lang_gen through decoder --> note that we need a causal mask on language generation...
        ctx_reconstructions, generated_language = self.forward_decoder(
            visible_ctx_patches, restore_idxs, lang_gen_embeddings, lang_gen_mask
        )

        # Compute loss for reconstructed patches & generated language
        loss, reconstruction_loss, lang_loss, zero_loss, k_loss = self.compute_loss(
            imgs, ctx_reconstructions, mask, lang_gen, generated_language, lang_gen_mask, lang_gen_weight
        )
        return loss, reconstruction_loss, lang_loss, [zero_loss, k_loss]

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
