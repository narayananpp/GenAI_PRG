import argparse
import os
import torch as th
import numpy as np
from torch.utils.data import DataLoader

from reward_model.model import create_reward_model
from reward_model.motion_datasets import load_motion_data
from train import run_eval, create_argparser
from reward_model.script_util import args_to_dict, reward_model_defaults
from utils.compute_physics_metrics import compute_physics_metrics

def evaluate():
    args = create_argparser().parse_args()
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")

    # 1. Load stats
    mean = np.load(args.mean_path)
    std = np.load(args.std_path)

    # 2. Dataset - Point specifically to the reward_samples subfolder
    # load_motion_data usually looks for pkls inside the path provided
    val_dataset = load_motion_data(
        args.data_dir, 
        mean, std, 
        max_len=args.max_seq_len, 
        split="val"
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )

    # 3. Model Creation - Filter args correctly to avoid TypeError
    model_keys = reward_model_defaults().keys()
    model_args = args_to_dict(args, model_keys)
    model = create_reward_model(**model_args)

    # 4. Load Checkpoint
    if args.resume_checkpoint:
        print(f"Loading checkpoint: {args.resume_checkpoint}")
        ckpt = th.load(args.resume_checkpoint, map_location=dev)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    
    model.to(dev).eval()

    # 5. Run Eval
    print(f"Samples found: {len(val_dataset)}")
    if len(val_dataset) == 0:
        print("Error: No validation samples found. Check your --data_dir")
        return

    try:
        loss, acc = run_eval(model, val_loader, dev, args)
        print("\n" + "="*30)
        print(f"VALIDATION ACCURACY: {acc*100:.2f}%")
        print(f"VALIDATION LOSS:     {loss:.4f}")
        print("="*30)
    except KeyError as e:
        print(f"Data Error: The dataset contains a file missing the key {e}.")
        print("Make sure --data_dir points to the folder containing 'reward_samples'")

if __name__ == "__main__":
    evaluate()