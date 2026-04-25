import os
import sys
import torch
import numpy as np
import pickle
from multiprocessing import Pool
from tqdm import tqdm

from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import train_args
from utils.compute_physics_metrics import compute_physics_metrics

CLEANED_DATA_DIR = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D"
MEAN_PATH        = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D/Mean.npy"
STD_PATH         = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D/Std.npy"

NUM_FEATURES  = 263
SPATIAL_DIM   = 1
CLIP_DENOISED = False
BATCH_SIZE    = 128

SAVE_FOLDER = "MDM_preds_GRPO"


def compute_final_metrics(x0_hat):
    if x0_hat.ndim == 2 and x0_hat.shape[0] == NUM_FEATURES and x0_hat.shape[1] != NUM_FEATURES:
        motion = x0_hat.T
    else:
        motion = x0_hat
    return compute_physics_metrics(motion)


def generate_and_harvest_trajectories(model, diffusion, cond, num_frames, mean_ten, std_ten):
    B     = next(iter(cond['y'].values())).shape[0]
    shape = (B, NUM_FEATURES, SPATIAL_DIM, num_frames)
    total_steps = diffusion.num_timesteps

    trajectory_dict = {}

    generator = diffusion.p_sample_loop_progressive(
        model,
        shape,
        model_kwargs=cond,
        clip_denoised=CLIP_DENOISED,
        progress=False,
        device=dist_util.dev(),
    )

    for i, sample in enumerate(generator):
        t         = (total_steps - 1) - i
        x0_raw    = sample['pred_xstart'].detach().cpu()
        x0_hat    = x0_raw[:, :, 0, :]
        x0_unnorm = x0_hat * std_ten + mean_ten
        trajectory_dict[t] = [
            x0_unnorm[b].permute(1, 0).numpy()
            for b in range(x0_unnorm.shape[0])
        ]

    assert 0 in trajectory_dict, "t=0 prediction missing"
    return trajectory_dict


def generate_mdm_dataset(model, diffusion, data, num_samples, num_frames, save_dir, run_idx):
    os.makedirs(os.path.join(save_dir, SAVE_FOLDER), exist_ok=True)

    mean_ten = torch.from_numpy(data.dataset.t2m_dataset.mean).float().view(1, -1, 1)
    std_ten  = torch.from_numpy(data.dataset.t2m_dataset.std ).float().view(1, -1, 1)

    name_list    = data.dataset.t2m_dataset.name_list
    samples_seen = 0

    for batch_idx, (motion_gt, cond) in enumerate(tqdm(data, desc=f"Run {run_idx}")):
        if samples_seen >= num_samples:
            break

        actual_bs = motion_gt.shape[0]
        lengths   = cond['y']['lengths']

        try:
            trajectory_dict = generate_and_harvest_trajectories(
                model, diffusion, cond, num_frames, mean_ten, std_ten
            )
        except Exception as e:
            print(f"[SKIP] Run {run_idx} failed for batch {batch_idx}: {e}")
            samples_seen += actual_bs
            continue

        for b in range(actual_bs):
            global_idx = batch_idx * BATCH_SIZE + b
            if global_idx >= len(name_list):
                break

            raw_name   = name_list[global_idx]
            actual_len = int(lengths[b].item())

            traj_b = {
                t: seqs[b][:actual_len]
                for t, seqs in trajectory_dict.items()
            }

            final_x0      = traj_b[0]
            final_metrics = compute_final_metrics(final_x0)

            save_path = os.path.join(save_dir, SAVE_FOLDER, f"{raw_name}_run{run_idx}.pkl")
            try:
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "name":          raw_name,
                        "run_idx":       run_idx,
                        "x0_trajectory": traj_b,
                        "final_metrics": final_metrics,
                    }, f)
            except Exception as e:
                print(f"[SKIP] Save failed for {raw_name} run {run_idx}: {e}")

        samples_seen += actual_bs

    print(f"Run {run_idx} done. Processed {min(samples_seen, num_samples)} samples.")


if __name__ == "__main__":
    import sys

    # remove --run_idx before train_args sees it
    run_idx = None
    for i, arg in enumerate(sys.argv):
        if arg == '--run_idx' and i + 1 < len(sys.argv):
            run_idx = int(sys.argv[i + 1])
            sys.argv.pop(i)
            sys.argv.pop(i)
            break
    assert run_idx is not None, "Must provide --run_idx"

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

    generate_mdm_dataset(
        model, diffusion, data,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        save_dir=args.save_dir,
        run_idx=run_idx,
    )