"""
Train a feasibility reward model using paired RLHF-style preference loss.

Each training step:
  1. Sample a batch of (positive, negative) pairs
  2. Forward both through the reward model
  3. Loss = -log σ(reward_pos - reward_neg)   [Bradley-Terry]

Multi-GPU: DistributedDataParallel (DDP) across N GPUs on a single node.
Usage:
    # Single GPU
    python train_reward_model.py --data_dir ... --mean_path ... --std_path ...

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 train_reward_model.py --data_dir ... --mean_path ... --std_path ...
"""

import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from reward_model import logger
from reward_model.model import create_reward_model
from reward_model.motion_datasets import load_motion_data, padding_mask
from reward_model.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    reward_model_defaults,
)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialise DDP if launched with torchrun, else single-GPU fallback."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        th.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ---------------------------------------------------------------------------
# Loss and accuracy
# ---------------------------------------------------------------------------

def paired_preference_loss(reward_pos, reward_neg):
    """
    Bradley-Terry paired preference loss.
    loss = -log σ(reward_pos - reward_neg)
    Minimising pushes reward_pos > reward_neg.
    """
    return -F.logsigmoid(reward_pos - reward_neg).mean()


def paired_accuracy(reward_pos, reward_neg):
    """Fraction of pairs where reward_pos > reward_neg."""
    return (reward_pos > reward_neg).float().mean().item()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(model, data, dev, args):
    total_loss = 0.0
    total_acc  = 0.0
    n = 0
    with th.no_grad():
        for motion_pos, motion_neg, length_pos, length_neg, t in data:
            motion_pos = motion_pos.to(dev, non_blocking=True)
            motion_neg = motion_neg.to(dev, non_blocking=True)
            length_pos = length_pos.to(dev, non_blocking=True)
            length_neg = length_neg.to(dev, non_blocking=True)
            t = t.to(dev, non_blocking=True) # Move t to device

            mask_pos = padding_mask(length_pos, args.max_seq_len)
            mask_neg = padding_mask(length_neg, args.max_seq_len)

            with autocast(enabled=args.amp):
                # Model now correctly uses t to modulate AdaLN blocks
                reward_pos = model(motion_pos, t, key_padding_mask=mask_pos)
                reward_neg = model(motion_neg, t, key_padding_mask=mask_neg)

            total_loss += paired_preference_loss(reward_pos, reward_neg).item()
            total_acc  += paired_accuracy(reward_pos, reward_neg)
            n += 1

    return total_loss / max(1, n), total_acc / max(1, n)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_model(model, opt, scheduler, scaler, step, args):
    os.makedirs(args.save_dir, exist_ok=True)
    # Unwrap DDP before saving
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    th.save(
        {
            "step":      step,
            "model":     model_state,
            "opt":       opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "config":    vars(args),
        },
        os.path.join(args.save_dir, f"model_{step:06d}.pt"),
    )


def load_checkpoint(model, opt, scheduler, scaler, path, dev):
    ckpt = th.load(path, map_location=dev)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()

    local_rank, rank, world_size = setup_ddp()
    dev = th.device(f"cuda:{local_rank}" if th.cuda.is_available() else "cpu")

    if is_main(rank):
        logger.configure()
        logger.log(f"World size: {world_size}")
        logger.log("Creating reward model...")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = create_reward_model(**args_to_dict(args, reward_model_defaults().keys()))
    model.to(dev)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main(rank):
        raw_model = model.module if hasattr(model, "module") else model
        logger.log(f"Parameters: {raw_model.count_parameters():,}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    if is_main(rank):
        logger.log("Loading dataset statistics...")

    mean = np.load(args.mean_path)   # (263,)
    std  = np.load(args.std_path)    # (263,)

    train_dataset = load_motion_data(
        args.data_dir, mean, std,
        max_len=args.max_seq_len,
        split="train",
    )
    val_dataset = load_motion_data(
        args.data_dir, mean, std,
        max_len=args.max_seq_len,
        split="val",
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False, drop_last=False) \
                    if world_size > 1 else None

    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    # ------------------------------------------------------------------ #
    # Optimizer, scheduler, scaler
    # ------------------------------------------------------------------ #
    opt       = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs * len(train_data), eta_min=1e-6)
    scaler    = GradScaler(enabled=args.amp)

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume_checkpoint:
        start_step = load_checkpoint(model, opt, scheduler, scaler, args.resume_checkpoint, dev)
        if is_main(rank):
            logger.log(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    if is_main(rank):
        logger.log("Training reward model...")

    step = start_step
    for epoch in range(args.epochs):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # ensures different shuffle per epoch with DDP

        model.train()
        for motion_pos, motion_neg, length_pos, length_neg, t in train_data:
            motion_pos = motion_pos.to(dev, non_blocking=True)
            motion_neg = motion_neg.to(dev, non_blocking=True)
            length_pos = length_pos.to(dev, non_blocking=True)
            length_neg = length_neg.to(dev, non_blocking=True)
            t = t.to(dev, non_blocking=True) # Move t to device

            mask_pos = padding_mask(length_pos, args.max_seq_len)
            mask_neg = padding_mask(length_neg, args.max_seq_len)

            with autocast(enabled=args.amp):
                # Model now correctly uses t to modulate AdaLN blocks
                reward_pos = model(motion_pos, t, key_padding_mask=mask_pos)
                reward_neg = model(motion_neg, t, key_padding_mask=mask_neg)
                loss = paired_preference_loss(reward_pos, reward_neg)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            acc = paired_accuracy(reward_pos.detach(), reward_neg.detach())

            if is_main(rank) and step % args.log_interval == 0:
                logger.logkv("step",       step)
                logger.logkv("epoch",      epoch)
                logger.logkv("loss",       loss.item())
                logger.logkv("acc",        acc)
                logger.logkv("lr",         scheduler.get_last_lr()[0])
                logger.logkv("reward_pos", reward_pos.mean().item())
                logger.logkv("reward_neg", reward_neg.mean().item())
                logger.dumpkvs()

            if is_main(rank) and step > 0 and step % args.eval_interval == 0:
                model.eval()
                try:
                    val_loss, val_acc = run_eval(model, val_data, dev, args)
                    logger.logkv("val_loss", val_loss)
                    logger.logkv("val_acc",  val_acc)
                    logger.dumpkvs()
                finally:
                    model.train()

            if is_main(rank) and step > 0 and step % args.save_interval == 0:
                logger.log(f"Saving checkpoint at step {step}...")
                save_model(model, opt, scheduler, scaler, step, args)

            step += 1

# 1. Ensure all ranks have finished the training loop
    if dist.is_initialized():
        dist.barrier()
        
    if is_main(rank):
        logger.log("Saving final model...")
        save_model(model, opt, scheduler, scaler, step, args)
        
    if dist.is_initialized():
            dist.barrier()

    cleanup_ddp()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def create_argparser():
    defaults = dict(
        # --- Data paths ---
        data_dir="",
        mean_path="data/Mean.npy",
        std_path="data/Std.npy",
        # --- Optimization ---
        epochs=4,
        lr=3e-4,
        weight_decay=1e-3,
        grad_clip=1.0,
        # --- Batching ---
        batch_size=32,
        num_workers=4,
        # --- AMP ---
        amp=True,                 # mixed precision — speeds up training ~2x on modern GPUs
        # --- Logging and saving ---
        log_interval=10,
        eval_interval=500,
        save_interval=2000,
        save_dir="checkpoints",
        resume_checkpoint="",     # path to .pt file to resume from
    )
    defaults.update(reward_model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()