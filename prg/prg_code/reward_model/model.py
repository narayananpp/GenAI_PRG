"""
Feasibility reward model for text-conditioned human motion sequences.

Architecture mirrors the classifier in Dhariwal & Nichol (2021): a
time-conditioned discriminative model trained across all noise levels,
using adaptive normalization (AdaGN / AdaLN) to inject the timestep.

For Approach 1 (no time conditioning), pass t=zeros(B).
For Approach 2 (time-conditioned),     pass the actual diffusion timestep.
"""

import math

import torch as th
import torch.nn as nn

from .nn import timestep_embedding, positional_embedding, zero_module


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into a vector representation.

    Uses the same sinusoidal schedule as MDM to ensure the t values seen
    at training time match those seen during guided inference.

    :param model_channels: width of the reward model.
    :param time_embed_dim: dimension of the sinusoidal frequency basis.
    """

    def __init__(self, model_channels, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

    def forward(self, timesteps):
        return self.mlp(timestep_embedding(timesteps, self.time_embed_dim))


class AdaLNBlock(nn.Module):
    """
    A transformer encoder block with AdaLN-Zero timestep conditioning.

    Follows the AdaGN design of Dhariwal & Nichol (2021), adapted to
    transformer layer norm. The modulation projection is zero-initialised
    so each block begins as an identity map, stabilising early training.

    :param channels: model width.
    :param num_heads: number of attention heads.
    :param ffn_mult: FFN hidden dim multiplier.
    :param dropout: dropout probability.
    """

    def __init__(self, channels, num_heads, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_mult * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * channels, channels),
        )
        self.drop = nn.Dropout(dropout)
        # Projects emb -> 6 x channels: (shift, scale, gate) for attn and ffn.
        # Zero-init so blocks start as identity (the "Zero" in AdaLN-Zero).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(channels, 6 * channels, bias=True)),
        )

    def forward(self, x, emb, key_padding_mask=None):
        """
        :param x: [B x T x C] input sequence.
        :param emb: [B x C] timestep embedding.
        :param key_padding_mask: [B x T] bool mask, True = padding.
        """
        s1, c1, g1, s2, c2, g2 = self.adaLN_modulation(emb).chunk(6, dim=-1)
        s1, c1, g1 = s1[:, None], c1[:, None], g1[:, None]
        s2, c2, g2 = s2[:, None], c2[:, None], g2[:, None]

        h = self.norm1(x) * (1.0 + c1) + s1
        h = h.transpose(0, 1)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        h = h.transpose(0, 1)
        x = x + g1 * self.drop(h)

        h = self.norm2(x) * (1.0 + c2) + s2
        x = x + g2 * self.drop(self.ffn(h))
        return x


class FeasibilityRewardModel(nn.Module):
    """
    Transformer encoder with CLS pooling that outputs a scalar feasibility
    reward for a HumanML3D motion sequence.

    Inputs:
        x : [B x T x 263]  normalised motion frames.
        t : [B]             diffusion timestep. Use zeros for clean x0.
        key_padding_mask : [B x T]  True = padding frame.

    Output:
        reward : [B]  scalar (higher = more physically plausible).

    :param in_channels: per-frame feature dimension (263 for HumanML3D).
    :param model_channels: transformer width.
    :param num_res_blocks: number of transformer blocks.
    :param num_heads: attention heads.
    :param ffn_mult: FFN hidden dim multiplier.
    :param dropout: dropout probability.
    :param max_seq_len: maximum motion clip length in frames.
    """

    def __init__(
        self,
        in_channels=263,
        model_channels=256,
        num_res_blocks=4,
        num_heads=4,
        ffn_mult=2,
        dropout=0.1,
        max_seq_len=196,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels

        self.frame_proj = nn.Linear(in_channels, model_channels)
        self.cls_token = nn.Parameter(th.zeros(1, 1, model_channels))
        self.register_buffer(
            "pos_emb", positional_embedding(max_seq_len + 1, model_channels)
        )

        self.time_embed = TimestepEmbedder(model_channels)

        self.blocks = nn.ModuleList(
            [AdaLNBlock(model_channels, num_heads, ffn_mult, dropout)
             for _ in range(num_res_blocks)]
        )
        self.out_norm = nn.LayerNorm(model_channels)
        # Zero-init final projection for stable early training.
        self.out_proj = nn.Sequential(
            nn.Linear(model_channels, model_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(model_channels // 2, 1)),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.frame_proj.weight)
        nn.init.zeros_(self.frame_proj.bias)

    def forward(self, x, t, key_padding_mask=None):
        b, seq_len, _ = x.shape

        h = self.frame_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        h = th.cat([cls, h], dim=1)
        h = h + self.pos_emb[:, : seq_len + 1, :]

        # Prepend False to mask so the CLS token is never masked out.
        if key_padding_mask is not None:
            cls_col = th.zeros(b, 1, dtype=th.bool, device=x.device)
            key_padding_mask = th.cat([cls_col, key_padding_mask], dim=1)

        emb = self.time_embed(t)
        for block in self.blocks:
            h = block(h, emb, key_padding_mask)

        out = self.out_norm(h[:, 0])        # CLS token
        return self.out_proj(out).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_reward_model(
    in_channels=263,
    model_channels=256,
    num_res_blocks=4,
    num_heads=4,
    ffn_mult=2,
    dropout=0.1,
    max_seq_len=196,
):
    """
    Instantiate a FeasibilityRewardModel from flat keyword arguments.
    Mirrors the create_classifier() factory in guided-diffusion.
    """
    return FeasibilityRewardModel(
        in_channels=in_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )
