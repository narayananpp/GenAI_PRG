"""
Score motion sequences with a trained feasibility reward model.

Two use cases:

    1. Standalone scoring / hard-negative selection — score a directory of
       MDM-inferred motion files and write out the bottom-quartile filenames.

    2. Gradient computation for guided diffusion — compute dR/dx0_hat at a
       given noise level, used inside the PRG guided sampler.
"""

import argparse
import os

import numpy as np
import torch as th

from reward_model import logger
from reward_model.model import FeasibilityRewardModel
from reward_model.motion_datasets import padding_mask
from reward_model.script_util import add_dict_to_argparser, reward_model_defaults


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_reward_model(checkpoint_path, device=None):
    """
    Restore a FeasibilityRewardModel from a checkpoint written by train.py.

    :param checkpoint_path: path to a .pt file saved by save_model().
    :param device: torch.device. Defaults to cuda if available.
    :return: (model, config_dict)
    """
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ckpt   = th.load(checkpoint_path, map_location=device)
    cfg    = ckpt["config"]
    model  = FeasibilityRewardModel(
        in_channels    = cfg.get("in_channels",    263),
        model_channels = cfg.get("model_channels", 256),
        num_res_blocks = cfg.get("num_res_blocks", 4),
        num_heads      = cfg.get("num_heads",      4),
        ffn_mult       = cfg.get("ffn_mult",       2),
        dropout        = 0.0,   # no dropout at inference
        max_seq_len    = cfg.get("max_seq_len",    196),
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, cfg


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@th.no_grad()
def score_motions(model, motions, mean, std, t=0, lengths=None, device=None):
    """
    Return a scalar feasibility reward for each motion in a batch.

    :param model: FeasibilityRewardModel in eval mode.
    :param motions: [B x T x 263] numpy array (unnormalised).
    :param mean: [263] dataset mean.
    :param std: [263] dataset std.
    :param t: int or [B] tensor. Diffusion timestep. Use 0 for clean x0.
    :param lengths: [B] actual sequence lengths. None = full length.
    :param device: torch.device.
    :return: [B] numpy array of reward scores.
    """
    if device is None:
        device = next(model.parameters()).device

    motions = th.tensor((motions - mean) / (std + 1e-8), dtype=th.float32).to(device)
    if motions.ndim == 2:
        motions = motions.unsqueeze(0)

    b, seq_len, _ = motions.shape

    if isinstance(t, int):
        t = th.full((b,), t, dtype=th.long, device=device)
    else:
        t = t.to(device)

    mask = None
    if lengths is not None:
        mask = padding_mask(lengths.to(device), seq_len)

    return model(motions, t, key_padding_mask=mask).cpu().numpy()


def select_hard_negatives(model, motion_dir, mean, std, device=None):
    """
    Score all .npy files in motion_dir and return the filenames in the
    bottom quartile by feasibility (highest infeasibility score).

    :param model: trained FeasibilityRewardModel.
    :param motion_dir: directory of MDM-inferred motion .npy files.
    :param mean: [263] dataset mean.
    :param std: [263] dataset std.
    :return: list of filenames (bottom 25% by reward).
    """
    fnames  = sorted(f for f in os.listdir(motion_dir) if f.endswith(".npy"))
    motions = np.stack([np.load(os.path.join(motion_dir, f)) for f in fnames])
    rewards = score_motions(model, motions, mean, std, device=device)
    threshold = np.percentile(rewards, 25)
    return [f for f, r in zip(fnames, rewards) if r <= threshold]


# ---------------------------------------------------------------------------
# Reward gradient (for guided diffusion sampler)
# ---------------------------------------------------------------------------

def reward_gradient(model, x0_hat, t, lengths=None):
    """
    Compute the reward and its gradient with respect to x0_hat.

    Used inside the PRG guided sampler:
        x0_hat += w(t) * grad

    :param model: FeasibilityRewardModel (weights stay frozen).
    :param x0_hat: [B x T x 263] predicted clean motion, on the model's device.
    :param t: [B] diffusion timestep tensor.
    :param lengths: [B] actual sequence lengths (optional).
    :return: (reward [B], grad [B x T x 263])
    """
    for p in model.parameters():
        p.requires_grad_(False)

    x = x0_hat.detach().requires_grad_(True)
    mask = None
    if lengths is not None:
        mask = padding_mask(lengths, x.shape[1]).to(x.device)

    reward = model(x, t, key_padding_mask=mask)
    reward.sum().backward()

    grad = x.grad.clone()
    return reward.detach(), grad


# ---------------------------------------------------------------------------
# CLI entry point (scoring / hard-negative selection)
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()
    logger.configure()

    logger.log("loading reward model...")
    model, _ = load_reward_model(args.model_path)

    mean = np.load(args.mean_path)
    std  = np.load(args.std_path)

    logger.log("selecting hard negatives...")
    hard_neg_fnames = select_hard_negatives(model, args.motion_dir, mean, std)

    out_path = os.path.join(args.motion_dir, "hard_negatives.txt")
    with open(out_path, "w") as f:
        for fname in hard_neg_fnames:
            f.write(fname + "\n")

    logger.log(f"wrote {len(hard_neg_fnames)} hard negatives to {out_path}")


def create_argparser():
    defaults = dict(
        model_path="",
        motion_dir="",
        mean_path="data/mean.npy",
        std_path="data/std.npy",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
