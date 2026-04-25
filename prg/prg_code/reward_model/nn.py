"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def positional_embedding(seq_len, dim):
    """
    Create sinusoidal positional embeddings for a sequence.

    :param seq_len: number of positions.
    :param dim: embedding dimension.
    :return: a [1 x seq_len x dim] Tensor.
    """
    pe = th.zeros(1, seq_len, dim)
    pos = th.arange(seq_len).unsqueeze(1).float()
    div = th.exp(th.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[0, :, 0::2] = th.sin(pos * div)
    pe[0, :, 1::2] = th.cos(pos * div)
    return pe


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
