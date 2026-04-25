# PRG: Physics-guided Reward Gradient for Human Motion Diffusion

This repository extends the [Human Motion Diffusion Model (MDM)](https://github.com/GuyTevet/motion-diffusion-model) with **physics-aware reward guidance** during the diffusion denoising process.

The core idea: train a reward model on paired motion trajectories ranked by physical plausibility, then use its gradients to steer the diffusion sampling toward more physically feasible motions — without any retraining of MDM itself.

---

## Overview

Standard MDM generates high-quality, text-aligned motions but does not explicitly optimize for physical constraints (foot contact, floor penetration, skating artifacts). PRG adds a lightweight guidance signal at inference time:

```
x_t
 └─► MDM (unconditional) ─┐
 └─► MDM (conditional)   ─┴─► CFG mean ─► + reward gradient ─► final mean
```

The reward gradient nudges each denoising step toward motions with lower physical error, measured by penetration, float, and skate metrics.

---

## Contributions

### 1. Reward Dataset Generation (`prg/generate_positive_samples.py`)

Generates paired preference data for reward model training:

- Runs MDM **N times** for each prompt in the HumanML3D dataset
- Computes physics metrics (`Phys_Err`, `Penetrate`, `Float`, `Skate`) for each run's final output
- Ranks runs by `Phys_Err` and saves the **best vs. worst trajectory** as a positive/negative pair
- Optionally includes **ground truth motion** as the positive when it outperforms all MDM runs
- Saves full denoising trajectories (`pred_xstart` at every timestep `t`) so the reward model learns to evaluate motions at any noise level

```bash
# Generate dataset (run 4 times with different seeds)
python prg/generate_positive_samples.py \
    --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
    --num_samples 1000 \
    --run_idx 0   # repeat for 1, 2, 3

# Build preference pairs
python prg/prg_code/build_reward_pairs.py
```

### 2. Reward Model Training (`prg/prg_code/train_reward_model.py`)

Trains a time-conditioned transformer reward model using Bradley-Terry preference loss:

```
loss = -log σ(reward_pos - reward_neg)
```

Key design choices:
- **Time conditioning via AdaLN**: the model takes diffusion timestep `t` as input, allowing it to evaluate motions relative to the expected noise level at each step
- **Multi-GPU DDP training** via `torchrun`
- **Mixed precision** (AMP) for faster training
- Paired preference loss encourages `reward(best_trajectory) > reward(worst_trajectory)` at every timestep

```bash
# Single GPU
python prg/prg_code/train_reward_model.py \
    --data_dir /path/to/reward_pairs \
    --mean_path dataset/HumanML3D/Mean.npy \
    --std_path dataset/HumanML3D/Std.npy \
    --save_dir checkpoints/

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 prg/prg_code/train_reward_model.py \
    --data_dir /path/to/reward_pairs \
    --mean_path dataset/HumanML3D/Mean.npy \
    --std_path dataset/HumanML3D/Std.npy
```

### 3. Reward-Guided Diffusion (`prg/prg_code/reward_model/reward_cond_fn.py`)

Plugs into MDM's classifier guidance interface (`condition_mean_with_grad`) to steer sampling:

- Computes `∇_{x0} reward(pred_xstart, t)` at each denoising step
- Normalizes gradients per-sample to prevent scale issues
- Applies a quadratic guidance schedule `((T-t)/T)²`
- Configurable `GUIDANCE_CUTOFF` to stop guiding below a timestep threshold (empirically, guiding only for `t ≥ 35` out of 50 steps works best)
- Full trajectory logging and reward plots saved automatically

```python
# In your eval script, pass reward_cond_fn as the cond_fn argument:
from prg.prg_code.reward_model.reward_cond_fn import reward_cond_fn

samples = diffusion.p_sample_loop(
    model,
    shape,
    model_kwargs=cond,
    cond_fn=reward_cond_fn,
    cond_fn_with_grad=True,
)
```

### 4. MDM Eval Integration (`data_loaders/humanml/motion_loaders/comp_v6_model_dataset_reward_gradient.py`)

Modified MDM's `CompMDMGeneratedDataset` to pass `reward_cond_fn` into the diffusion sampling loop:

```python
sample = sample_fn(
    model,
    motion.shape,
    clip_denoised=clip_denoised,
    model_kwargs=model_kwargs,
    cond_fn=reward_cond_fn,       # ← reward gradient injected here
    cond_fn_with_grad=True,
)
```

This is the only change needed to MDM's eval pipeline — the reward gradient is applied on top of CFG at every denoising step automatically. The file is a drop-in replacement for the original `comp_v6_model_dataset.py`.

---

## Setup

Follow the [MDM setup instructions](https://github.com/GuyTevet/motion-diffusion-model#getting-started) first, then:

```bash
# No additional dependencies required beyond MDM's environment
conda activate mdm

# Download reward model checkpoint (if provided)
# Place in: checkpoints/model_010668.pt
```

Update the checkpoint path in `reward_cond_fn.py`:
```python
ckpt = torch.load("checkpoints/model_010668.pt", map_location="cuda:0")
```

---

## Evaluation

Run the standard MDM eval with reward guidance enabled:

```bash
python -m eval.eval_humanml \
    --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
    --device 0
```

Physics metrics are logged automatically at the end of each replication:

```
========== Physics Metrics [vald] ==========
  Penetrate   :   14.29 +/- 131.6  mm
  Float       :   26.35 +/-  70.0  mm
  Skate       :    9.54 +/-  15.7  mm
  Phys_Err    :   50.19 +/- 148.3  mm
  Reference (MDM, PhysDiff Table 1):
    Penetrate=11.291  Float=18.876  Skate=1.406  Phys_Err=31.572
```

Reward trajectory plots are saved to `reward_analysis/` after each generation.

---

## Results

| Method | Phys_Err ↓ | FID ↓ | R-precision (top-1) ↑ |
|---|---|---|---|
| MDM (baseline) | 50.19 | 0.419 | 0.392 |
| MDM + PRG (ours) | ~50.1 | 0.429 | 0.421 |
| PhysDiff (reference) | 31.57 | — | — |

> **Note:** Current results show marginal improvement. The reward model is undertrained (10k steps) and further training with larger, more diverse preference data is expected to improve Phys_Err meaningfully.

---

## Project Structure

```
prg/
├── datagen.py                        # Core data generation logic
├── datagen_grpo_style.py             # GRPO-style multi-run data generation
├── generate_positive_samples.py      # MDM rollout + physics scoring (positive samples)
├── generate_rm_dataset.py            # Full reward model dataset generation
├── launch_datagen_grpo.sh            # Shell launcher for multi-run data generation
├── test_grpo_style.py                # Tests for GRPO-style datagen
├── test_gt.py                        # Tests using ground truth motions
├── reward_dataset/                   # Generated preference pair data (not tracked)
└── prg_code/
    ├── train.py                      # Reward model training loop
    ├── eval.py                       # Reward model evaluation
    ├── score_motions.py              # Score motions with trained reward model
    ├── dummy_run.py                  # Sanity check / quick test run
    ├── README.md                     # PRG-specific notes
    └── reward_model/
        ├── __init__.py
        ├── model.py                  # Time-conditioned transformer reward model
        ├── motion_datasets.py        # Paired motion dataset loader
        ├── reward_cond_fn.py         # Guidance function + analysis logging
        ├── nn.py                     # Neural network building blocks
        ├── logger.py                 # Logging utilities
        └── script_util.py            # Model defaults and arg helpers

data_loaders/humanml/motion_loaders/
└── comp_v6_model_dataset_reward_gradient.py  # Modified MDM eval dataset
                                               # with reward_cond_fn injected
                                               # into p_sample_loop
```

---

## Acknowledgments

This work builds directly on top of:

- **[MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model)** (Tevet et al., ICLR 2023) — the base diffusion model, training/eval pipeline, and classifier guidance interface used for reward steering.
- **[PhysDiff](https://github.com/ZhengyiLuo/PhysDiff)** — physics metrics (`Penetrate`, `Float`, `Skate`, `Phys_Err`) used for reward labeling and evaluation.
- **[guided-diffusion](https://github.com/openai/guided-diffusion)** — classifier guidance implementation that MDM and this work build on.

Please cite MDM if you use this code:

```bibtex
@inproceedings{tevet2023human,
  title={Human Motion Diffusion Model},
  author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}
```

---

## License

This code inherits the [MIT License](LICENSE) from MDM. Physics metric utilities may carry their own licenses — please refer to their respective repositories.