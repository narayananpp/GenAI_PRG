import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import os

from prg.prg_code.reward_model.model import create_reward_model

# ---------------------------------------------------------------------------
# Load reward model
# ---------------------------------------------------------------------------

reward_model = create_reward_model()
ckpt = torch.load(
    "/home/npalghat/projects/GenAI/motion-diffusion-model/checkpoints/model_010668.pt",
    map_location="cuda:4"
)
reward_model.load_state_dict(ckpt["model"])
reward_model.to("cuda:4")
reward_model.eval()

# ---------------------------------------------------------------------------
# Guidance schedule
# ---------------------------------------------------------------------------

T = 50  # num denoising steps

def guidance_scale_fn(t):
    t_val = t[0].item() if hasattr(t, "__len__") else float(t)
    return ((T - t_val) / T) ** 2

# ---------------------------------------------------------------------------
# Analysis state (accumulated across denoising steps)
# ---------------------------------------------------------------------------

_analysis = {
    "timesteps":    [],
    "rewards_mean": [],
    "rewards_std":  [],
    "grad_norms":   [],
    "scales":       [],
}

def reset_analysis():
    for v in _analysis.values():
        v.clear()

def save_analysis(out_dir="reward_analysis"):
    os.makedirs(out_dir, exist_ok=True)

    steps  = _analysis["timesteps"]
    if not steps:
        print("[reward_cond_fn] No analysis data to save.")
        return

    # --- 1. Reward over timesteps ---
    plt.figure()
    plt.plot(steps, _analysis["rewards_mean"], label="reward mean")
    plt.fill_between(
        steps,
        [m - s for m, s in zip(_analysis["rewards_mean"], _analysis["rewards_std"])],
        [m + s for m, s in zip(_analysis["rewards_mean"], _analysis["rewards_std"])],
        alpha=0.3, label="±1 std"
    )
    plt.xlabel("Timestep t")
    plt.ylabel("Reward")
    plt.title("Reward over denoising trajectory")
    plt.legend()
    plt.gca().invert_xaxis()  # t goes 49 → 0
    plt.savefig(os.path.join(out_dir, "reward_trajectory.png"))
    plt.close()

    # --- 2. Gradient norm over timesteps ---
    plt.figure()
    plt.plot(steps, _analysis["grad_norms"])
    plt.xlabel("Timestep t")
    plt.ylabel("Grad norm (pre-normalization)")
    plt.title("Gradient norm over denoising trajectory")
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(out_dir, "grad_norm_trajectory.png"))
    plt.close()

    # --- 3. Guidance scale over timesteps ---
    plt.figure()
    plt.plot(steps, _analysis["scales"])
    plt.xlabel("Timestep t")
    plt.ylabel("Guidance scale")
    plt.title("Guidance scale schedule")
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(out_dir, "guidance_scale.png"))
    plt.close()

    print(f"[reward_cond_fn] Analysis saved to {out_dir}/")
    print(f"  Final reward (t=0):   {_analysis['rewards_mean'][-1]:.4f}")
    print(f"  Initial reward (t=T): {_analysis['rewards_mean'][0]:.4f}")
    delta = _analysis['rewards_mean'][-1] - _analysis['rewards_mean'][0]
    print(f"  Reward delta:         {delta:+.4f}  ({'improving' if delta > 0 else 'degrading'})")

# ---------------------------------------------------------------------------
# Conditioning function
# ---------------------------------------------------------------------------

GUIDANCE_CUTOFF = 35

def reward_cond_fn(x, t, p_mean_var=None, **model_kwargs):
    t_val = t[0].item() if hasattr(t, "__len__") else float(t)

    # Trigger save at end of trajectory regardless of cutoff
    if t_val == 0:
        save_analysis()
        reset_analysis()
        return torch.zeros_like(x)

    # Stop guiding below cutoff
    if t_val < GUIDANCE_CUTOFF:
        return torch.zeros_like(x)

    # --- Prepare x0 ---
    x0 = p_mean_var["pred_xstart"]
    if x0.ndim == 4:
        x0 = x0[:, :, 0, :]
    x0 = x0.permute(0, 2, 1)
    x0 = x0.to(next(reward_model.parameters()).device).detach().requires_grad_(True)

    # --- Forward ---
    reward = reward_model(x0, t)

    # --- Gradient ---
    grad = torch.autograd.grad(reward.sum(), x0)[0]

    # --- NaN / Inf guards ---
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print(f"[reward_cond_fn] WARNING: NaN/Inf in gradient at t={t_val:.0f}, returning zero grad")
        return torch.zeros_like(x)

    if torch.isnan(reward).any():
        print(f"[reward_cond_fn] WARNING: NaN in reward at t={t_val:.0f}")

    # --- Logging ---
    B = grad.shape[0]
    grad_norm_val = grad.reshape(B, -1).norm(dim=1).mean().item()
    scale = guidance_scale_fn(t) * 10

    print(f"t={t_val:5.1f} | "
          f"reward={reward.mean().item():+.4f} ± {reward.std().item():.4f} | "
          f"grad_norm={grad_norm_val:.6f} | "
          f"scale={scale:.4f}")

    # --- Accumulate ---
    _analysis["timesteps"].append(t_val)
    _analysis["rewards_mean"].append(reward.mean().item())
    _analysis["rewards_std"].append(reward.std().item())
    _analysis["grad_norms"].append(grad_norm_val)
    _analysis["scales"].append(scale)

    # --- Normalize gradient ---
    grad_flat = grad.reshape(B, -1)
    grad_norm = grad_flat.norm(dim=1, keepdim=True) + 1e-8
    grad = (grad_flat / grad_norm).reshape(grad.shape)
    grad = grad.permute(0, 2, 1).unsqueeze(2)

    return scale * grad