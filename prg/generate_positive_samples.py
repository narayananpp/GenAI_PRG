import os
import torch
import numpy as np
import pickle
from tqdm import tqdm

from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import train_args

CLEANED_DATA_DIR = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D"
MEAN_PATH        = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D/Mean.npy"
STD_PATH         = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D/Std.npy"

NUM_FEATURES  = 263
SPATIAL_DIM   = 1
CLIP_DENOISED = False
BATCH_SIZE    = 128

SAVE_FOLDER = "GT_preds"


def harvest_gt_conditioned_trajectories(model, diffusion, cond,
                                         motion_gt_norm, num_frames,
                                         mean_ten, std_ten):
    x0_gt       = motion_gt_norm.to(dist_util.dev())   # (B, 263, 1, num_frames)
    B           = x0_gt.shape[0]
    total_steps = diffusion.num_timesteps

    trajectory_dict = {}

    for t_int in range(total_steps):
        t_tensor = torch.tensor([t_int] * B, device=dist_util.dev())

        noise = torch.randn_like(x0_gt)
        x_t   = diffusion.q_sample(x_start=x0_gt, t=t_tensor, noise=noise)

        with torch.no_grad():
            out = diffusion.p_mean_variance(
                model,
                x_t,
                t_tensor,
                model_kwargs=cond,
                clip_denoised=CLIP_DENOISED,
            )

        x0_hat    = out['pred_xstart'].detach().cpu()          # (B, 263, 1, num_frames)
        x0_hat    = x0_hat[:, :, 0, :]                        # (B, 263, num_frames)
        x0_unnorm = x0_hat * std_ten + mean_ten                # (B, 263, num_frames)

        trajectory_dict[t_int] = [
            x0_unnorm[b].permute(1, 0).numpy()                # (num_frames, 263)
            for b in range(B)
        ]

    assert 0 in trajectory_dict, "t=0 prediction missing"
    return trajectory_dict


def generate_gt_dataset(model, diffusion, data, num_samples, num_frames, save_dir):
    os.makedirs(os.path.join(save_dir, SAVE_FOLDER), exist_ok=True)

    mean_ten = torch.from_numpy(data.dataset.t2m_dataset.mean).float().view(1, -1, 1)
    std_ten  = torch.from_numpy(data.dataset.t2m_dataset.std ).float().view(1, -1, 1)

    name_list    = data.dataset.t2m_dataset.name_list
    samples_seen = 0

    for batch_idx, (motion_gt, cond) in enumerate(tqdm(data, desc="Generating GT")):
        if samples_seen >= num_samples:
            break

        actual_bs = motion_gt.shape[0]
        lengths   = cond['y']['lengths']   # (B,) actual frame counts

        try:
            trajectory_dict = harvest_gt_conditioned_trajectories(
                model, diffusion, cond,
                motion_gt, num_frames, mean_ten, std_ten,
            )
        except Exception as e:
            print(f"[SKIP] Generation failed for batch {batch_idx}: {e}")
            samples_seen += actual_bs
            continue

        for b in range(actual_bs):
            if samples_seen >= num_samples:
                break

            global_idx = batch_idx * BATCH_SIZE + b
            if global_idx >= len(name_list):
                break

            raw_name   = name_list[global_idx]
            actual_len = int(lengths[b].item())

            traj_b = {
                t: seqs[b][:actual_len]
                for t, seqs in trajectory_dict.items()
            }

            save_path = os.path.join(save_dir, SAVE_FOLDER, f"{raw_name}.pkl")
            try:
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "name":          raw_name,
                        "x0_trajectory": traj_b,
                    }, f)
            except Exception as e:
                print(f"[SKIP] Save failed for {raw_name}: {e}")

            samples_seen += 1

    print(f"Done. Saved {samples_seen} samples to {os.path.join(save_dir, SAVE_FOLDER)}")


if __name__ == "__main__":
    args = train_args()
    dist_util.setup_dist(args.device)

    args.data_dir = CLEANED_DATA_DIR
    mean_npy = np.load(MEAN_PATH)
    std_npy  = np.load(STD_PATH)

    data = get_dataset_loader(
        name=args.dataset,
        batch_size=BATCH_SIZE,
        num_frames=args.num_frames,
        split='cleaned_gt',
        hml_mode='train',
        device=dist_util.dev(),
    )

    data.dataset.t2m_dataset.mean = mean_npy
    data.dataset.t2m_dataset.std  = std_npy

    if data.sampler.__class__.__name__ == 'RandomSampler':
        from torch.utils.data import DataLoader, SequentialSampler
        data = DataLoader(
            data.dataset,
            batch_size=BATCH_SIZE,
            sampler=SequentialSampler(data.dataset),
            num_workers=data.num_workers,
            collate_fn=data.collate_fn,
            drop_last=False,
        )
        print("Replaced RandomSampler with SequentialSampler.")

    model, diffusion = create_model_and_diffusion(args, data)
    print(f"Loading checkpoint: {args.model_path}")
    load_saved_model(model, args.model_path)
    model.to(dist_util.dev())
    model.eval()

    generate_gt_dataset(
        model, diffusion, data,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        save_dir=args.save_dir,
    )