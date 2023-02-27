"""
vr3m.py

PyTorch Module defining an R3M model (with a ViT encoder), with the remainder as described in Nair et. al. 2021,
with all the requisite hyperparameters.

Reference:
    - https://github.com/facebookresearch/r3m
"""
from typing import Callable, Tuple

import torch
import torch.nn as nn
import transformers
from einops import rearrange

from voltron.models.util.optimization import get_lr_update
from voltron.models.util.transformer import Block, PatchEmbed, get_2D_position_embeddings

# Suppress Transformers Logging
transformers.logging.set_verbosity_error()


class VR3M(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        depth: int,
        embed_dim: int,
        n_heads: int,
        language_model: str,
        hf_cache: str,
        language_dim: int,
        reward_dim: int,
        n_negatives: int,
        lang_reward_weight: float,
        tcn_weight: float,
        l1_weight: float,
        l2_weight: float,
        optimizer: str,
        schedule: str,
        lr: float,
        min_lr: float,
        warmup_epochs: int,
        max_epochs: int,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        eps: float = 1e-8,
    ):
        """
        Initialize a ViT R3M model with the requisite architecture parameters.

        :param resolution: Base image resolution -- usually 224 (ImageNet size).
        :param patch_size: Height/Width of each patch in pixels -- usually 16.
        :param depth: Number of Transformer blocks in the ViT image encoder.
        :param embed_dim: Core embedding/hidden dimension for the vision transformer backbone.
        :param n_heads: Number of heads for multi-headed self-attention.
        :param language_model: Language model to freeze for encoding narrations/utterances.
        :param hf_cache: Cache directory to store pretrained models, for safe distributed training.
        :param language_dim: Dimensionality of the language embedding coming out of the pretrained LM.
        :param reward_dim: Hidden layer dimensionality for the language-reward MLP.
        :param n_negatives: Number of cross-batch negatives to sample for contrastive learning.
        :param lang_reward_weight: Weight applied to the contrastive "language alignment" loss term.
        :param tcn_weight: Weight applied to the time contrastive loss term.
        :param l1_weight: Weight applied to the L1 regularization loss term.
        :param l2_weight: Weight applied to the L2 regularization loss term.
        :param optimizer: String denoting which optimizer to use (for R3M, usually `adam`)
        :param schedule: Learning rate schedule to use; for Transformers a linear warmup + decay is recommended!
        :param lr: Peak learning rate to use over the course of training -- warms up to this value, then decays.
        :param min_lr: Minimum learning rate to decay to over the course of learning (usually 0.0)
        :param warmup_epochs: Number of epochs to warmup learning rate for linear warmup schedule.
        :param max_epochs: Total number of training epochs to be run.
        :param mlp_ratio: Ratio for embedding size to Position-wise FeedForward MLP (gets shrunk back down; default 4).
        :param in_channels: Default number of channels in the base image -- almost always 3.
        :param eps: Epsilon for preventing divide by zero in the InfoNCE loss terms.
        """
        super().__init__()
        self.resolution, self.patch_size, self.n_negatives, self.eps = resolution, patch_size, n_negatives, eps
        self.optimizer, self.schedule, self.lr, self.min_lr = optimizer, schedule, lr, min_lr
        self.warmup_epochs, self.max_epochs = warmup_epochs, max_epochs
        self.mlp_ratio, self.in_channels = mlp_ratio, in_channels
        self.language_dim, self.reward_dim = language_dim, reward_dim

        # Weights for each loss term
        self.lang_reward_weight, self.tcn_weight = lang_reward_weight, tcn_weight
        self.l1_weight, self.l2_weight = l1_weight, l2_weight

        # ViT Backbone
        self.depth, self.embed_dim, self.n_heads = depth, embed_dim, n_heads

        # Create ViT --> some differences from the original architectures in "An Image is Worth 16x16 Words":
        #   > Namely, subset of authors show that (https://arxiv.org/abs/2205.01580):
        #       1) 2D sinusoidal embeddings are better than learned embeddings...
        #       2) Average pooling is better/simpler than learning a [CLS] token embedding...
        # Note: Output "embedding" is just mean pooled final transformer layer --> after extra layer norm!
        #   > Ref: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L446
        self.patch2embed = PatchEmbed(self.resolution, self.patch_size, self.embed_dim, in_channels=self.in_channels)
        self.pe = nn.Parameter(torch.zeros(1, self.patch2embed.num_patches, self.embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(self.embed_dim, self.n_heads, self.mlp_ratio) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # Create Language Reward Model
        self.language_reward = nn.Sequential(
            nn.Linear(self.embed_dim + self.embed_dim + self.language_dim, self.reward_dim),
            nn.ReLU(),
            nn.Linear(self.reward_dim, self.reward_dim),
            nn.ReLU(),
            nn.Linear(self.reward_dim, self.reward_dim),
            nn.ReLU(),
            nn.Linear(self.reward_dim, self.reward_dim),
            nn.ReLU(),
            nn.Linear(self.reward_dim, 1),
            nn.Sigmoid(),
        )

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
        for _name, param in self.lm.named_parameters():
            param.requires_grad = False

    def initialize_weights(self) -> None:
        # Position Encoding -- Fixed 2D Sine-Cosine Embeddings
        pe = get_2D_position_embeddings(self.embed_dim, int(self.patch2embed.num_patches**0.5))
        self.pe.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # Initialize PatchEmbedding as a Linear...
        nn.init.xavier_uniform_(self.patch2embed.proj.weight.data.view([self.patch2embed.proj.weight.data.shape[0], -1]))

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

    def get_representations(self, img: torch.Tensor) -> torch.Tensor:
        """
        Given a single image, extract R3M "default" (mean-pooled) dense representation.

        :param img: Processed batch of images :: [bsz, 3, 224, 224]
        :return: Extracted R3M dense representation given img input.
        """
        assert img.ndim == 4, "Invalid input to `get_representations()`"

        patches = self.patch2embed(img)
        img_embedding = patches + self.pe

        # Apply Transformer Blocks...
        for block in self.blocks:
            img_embedding = block(img_embedding)
        img_embedding = self.norm(img_embedding)
        return img_embedding.mean(1, keepdim=True)

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """Feed images through ViT, get single embedding via global mean pooling."""
        patches = self.patch2embed(imgs)
        img_embedding = patches + self.pe

        # Apply Transformer Blocks...
        for block in self.blocks:
            img_embedding = block(img_embedding)
        img_embedding = self.norm(img_embedding)

        # Return mean pooled embeddings
        return img_embedding.mean(dim=1)

    def encode_language(self, lang: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """Encode language by feeding the *pre-tokenized text* through the frozen language model."""
        self.lm.eval()
        with torch.no_grad():
            transformer_embeddings = self.lm(lang, attention_mask=lang_mask).last_hidden_state
            return transformer_embeddings.mean(dim=1)

    def get_reward(self, initial: torch.Tensor, later: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        return self.language_reward(torch.cat([initial, later, lang], dim=-1)).squeeze()

    def forward(self, imgs: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor) -> Tuple[torch.tensor, ...]:
        """
        Run a forward pass through the model, computing the *full* R3M loss -- the TCN contrastive loss, the Language
        Alignment loss, and both sparsity losses, as well as the full loss (which will get optimized)!

        :param imgs: A [bsz, 5, in_channels, resolution, resolution] tensor of (start, i, j, k, end) sequences.
        :param lang: Tokenized language of dimensionality [bsz, seq_len] to be fed to the language model.
        :param lang_mask: Attention mask computed by the tokenizer, as a result of padding to the max_seq_len.

        :return: Tuple of losses, as follows:
            > (combined_loss, tcn_loss, reward_loss, l1_loss, l2_loss, tcn_acc, reward_acc)
        """
        # Encode each image separately... feed to transformer... then reshape
        all_images = rearrange(imgs, "bsz n_states c res1 res2 -> (bsz n_states) c res1 res2", n_states=5)
        all_embeddings = self.encode_images(all_images)
        initial, state_i, state_j, state_k, final = rearrange(
            all_embeddings, "(bsz n_states) embed -> n_states bsz embed", n_states=5
        )

        # Compute Regularization Losses
        l1_loss = torch.linalg.norm(all_embeddings, ord=1, dim=-1).mean()
        l2_loss = torch.linalg.norm(all_embeddings, ord=2, dim=-1).mean()

        # Compute TCN Loss
        tcn_loss, tcn_acc = self.get_time_contrastive_loss(state_i, state_j, state_k)

        # Compute Language Alignment/Predictive Loss
        lang_reward_loss, rew_acc = self.get_reward_loss(lang, lang_mask, initial, state_i, state_j, state_k, final)

        # Compute full weighted loss & return...
        loss = (
            (self.l1_weight * l1_loss)
            + (self.l2_weight * l2_loss)
            + (self.tcn_weight * tcn_loss)
            + (self.lang_reward_weight * lang_reward_loss)
        )
        return loss, tcn_loss, lang_reward_loss, l1_loss, l2_loss, tcn_acc, rew_acc

    @staticmethod
    def time_similarity(state_x: torch.Tensor, state_y: torch.Tensor, use_l2: bool = True) -> torch.Tensor:
        """Computes similarity between embeddings via -L2 distance."""
        assert use_l2, "Non-L2 time-similarity functions not yet implemented!"
        return -torch.linalg.norm(state_x - state_y, dim=-1)

    def get_time_contrastive_loss(
        self, state_i: torch.Tensor, state_j: torch.Tensor, state_k: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Evaluates the Time-Contrastive Loss, computed via InfoNCE."""

        # *Punchline* - we want `sim(i, j)` to be higher than `sim(i, k)` for some k > j (goes both ways)
        # `Reward(s*_0, s*_<t, l)` where * indicates a different video...
        #       > As our positive examples --> we sample (s_i, s_j) and (s_j, s_k).
        #       > Our negatives --> other pairs from the triplet, cross-batch negatives!
        sim_i_j_exp = torch.exp(self.time_similarity(state_i, state_j))
        sim_j_k_exp = torch.exp(self.time_similarity(state_j, state_k))

        # Add a "hard" negative!
        neg_i_k_exp = torch.exp(self.time_similarity(state_i, state_k))

        # Obtain *cross-batch* negatives
        bsz, neg_i, neg_j = state_i.shape[0], [], []
        for _ in range(self.n_negatives):
            neg_idx = torch.randperm(bsz)
            state_i_shuf = state_i[neg_idx]
            neg_idx = torch.randperm(bsz)
            state_j_shuf = state_j[neg_idx]
            neg_i.append(self.time_similarity(state_i, state_i_shuf))
            neg_j.append(self.time_similarity(state_j, state_j_shuf))
        neg_i_exp, neg_j_exp = torch.exp(torch.stack(neg_i, -1)), torch.exp(torch.stack(neg_j, -1))

        # Compute InfoNCE
        denominator_i = sim_i_j_exp + neg_i_k_exp + neg_i_exp.sum(-1)
        denominator_j = sim_j_k_exp + neg_i_k_exp + neg_j_exp.sum(-1)
        nce_i = -torch.log(self.eps + (sim_i_j_exp / (self.eps + denominator_i)))
        nce_j = -torch.log(self.eps + (sim_j_k_exp / (self.eps + denominator_j)))
        nce = (nce_i + nce_j) / 2

        # Compute "accuracy"
        i_j_acc = (1.0 * (sim_i_j_exp > neg_i_k_exp)).mean()
        j_k_acc = (1.0 * (sim_j_k_exp > neg_i_k_exp)).mean()
        acc = (i_j_acc + j_k_acc) / 2

        return nce.mean(), acc

    def get_reward_loss(
        self,
        lang: torch.Tensor,
        lang_mask: torch.Tensor,
        initial: torch.Tensor,
        state_i: torch.Tensor,
        state_j: torch.Tensor,
        state_k: torch.Tensor,
        final: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Evaluates the Language-Alignment Reward Loss, computed via InfoNCE."""
        lang_embed = self.encode_language(lang, lang_mask)

        # *Punchline* - we want `Reward(s_0, s_t, l)` to be higher than `Reward(s_0, s_<t, l)` AND
        # `Reward(s*_0, s*_<t, l)` where * indicates a different video...
        #       > As our positive examples --> we sample s_j, s_k, and s_final (excluding s_i)
        pos_final_exp = torch.exp(self.get_reward(initial, final, lang_embed))
        pos_j_exp = torch.exp(self.get_reward(initial, state_j, lang_embed))
        pos_k_exp = torch.exp(self.get_reward(initial, state_k, lang_embed))

        # Add the within-context negatives <--> these are the most informative examples!
        #   > We use initial, initial as a negative for the first one, just to get reward model to "capture progress"
        negs_final = [self.get_reward(initial, initial, lang_embed)]
        negs_j = [self.get_reward(initial, state_i, lang_embed)]
        negs_k = [self.get_reward(initial, state_j, lang_embed)]

        # Cross Batch Negatives -- same as positives (indexing), but from a different batch!
        #   > @SK :: Unclear how well this will unroll on TPUs...
        bsz = initial.shape[0]
        for _ in range(self.n_negatives):
            # We get three random indices to further minimize correlation... from the R3M codebase!
            neg_idx = torch.randperm(bsz)
            negs_final.append(self.get_reward(initial[neg_idx], final[neg_idx], lang_embed))
            neg_idx = torch.randperm(bsz)
            negs_j.append(self.get_reward(initial[neg_idx], state_j[neg_idx], lang_embed))
            neg_idx = torch.randperm(bsz)
            negs_k.append(self.get_reward(initial[neg_idx], state_k[neg_idx], lang_embed))

        # Flatten & exponentiate; get ready for the InfoNCE
        negs_final, negs_j, negs_k = torch.stack(negs_final, -1), torch.stack(negs_j, -1), torch.stack(negs_k, -1)
        negs_final_exp, negs_j_exp, negs_k_exp = torch.exp(negs_final), torch.exp(negs_j), torch.exp(negs_k)

        # Compute InfoNCE
        denominator_final = pos_final_exp + negs_final_exp.sum(-1)
        denominator_j = pos_j_exp + negs_j_exp.sum(-1)
        denominator_k = pos_k_exp + negs_k_exp.sum(-1)

        nce_final = -torch.log(self.eps + (pos_final_exp / (self.eps + denominator_final)))
        nce_j = -torch.log(self.eps + (pos_j_exp / (self.eps + denominator_j)))
        nce_k = -torch.log(self.eps + (pos_k_exp / (self.eps + denominator_k)))

        # Compute "accuracy"
        acc_final = (1.0 * (negs_final_exp.max(dim=-1)[0] < pos_final_exp)).mean()
        acc_j = (1.0 * (negs_j_exp.max(dim=-1)[0] < pos_j_exp)).mean()
        acc_k = (1.0 * (negs_k_exp.max(dim=-1)[0] < pos_k_exp)).mean()
        acc = (acc_final + acc_j + acc_k) / 3
        nce = (nce_final + nce_j + nce_k) / 3

        return nce.mean(), acc

    def configure_optimizer(self) -> Tuple[torch.optim.Optimizer, Callable[[int, float], float]]:
        # Short-Circuit on Valid Optimizers
        if self.optimizer not in ["adam"]:
            raise NotImplementedError(f"Optimizer `{self.optimizer}` not supported - try [`adam`] instead!")

        # Create Optimizer & LR Scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        update_lr = get_lr_update(optimizer, self.schedule, self.lr, self.min_lr, self.warmup_epochs, self.max_epochs)
        return optimizer, update_lr
