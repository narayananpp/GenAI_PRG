import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, "/home/npalghat/projects/GenAI/motion-diffusion-model")
from utils.compute_physics_metrics import compute_physics_metrics

# ====================================================================== #
#  CONFIGURATION
# ====================================================================== #
GRPO_DIR   = "/data3/npalghat/reward_dataset/MDM_preds_GRPO"
GT_DIR     = "/data3/npalghat/reward_dataset/GT_preds"
OUTPUT_DIR = "/data3/npalghat/reward_dataset/reward_pairs"

NUM_RUNS   = 4
# ====================================================================== #

def load_run(args):
    raw_name, run_idx, grpo_dir = args
    paths = [os.path.join(grpo_dir, f"{raw_name}_run{run_idx}.pkl")]
    if run_idx == 0:
        paths.append(os.path.join(grpo_dir, f"{raw_name}.pkl"))
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return run_idx, pickle.load(f)
            except Exception:
                return run_idx, None
    return run_idx, None

def load_gt(args):
    raw_name, gt_dir = args
    path = os.path.join(gt_dir, f"{raw_name}.pkl")
    if not os.path.exists(path):
        return raw_name, None
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        final_x0  = d["x0_trajectory"][0]
        metrics   = compute_physics_metrics(final_x0)
        return raw_name, d["x0_trajectory"], metrics
    except Exception:
        return raw_name, None, None

def get_all_prompt_names(grpo_dir):
    names = set()
    for fname in os.listdir(grpo_dir):
        if not fname.endswith(".pkl"): continue
        base = fname.replace(".pkl", "")
        for r in range(NUM_RUNS):
            if base.endswith(f"_run{r}"):
                base = base[:-(len(f"_run{r}"))]
                break
        names.add(base)
    return sorted(names)

def save_to_disk(name, t, pos_traj, neg_traj, pos_metrics, neg_metrics, pos_source):
    save_path = os.path.join(OUTPUT_DIR, f"{name}_t{t}_{pos_source}.pkl")
    try:
        with open(save_path, "wb") as f:
            pickle.dump({
                "name": name, "t": t,
                "positive_frame": pos_traj[t], "negative_frame": neg_traj[t],
                "positive_metrics": pos_metrics, "negative_metrics": neg_metrics,
                "positive_source": pos_source
            }, f)
    except Exception as e:
        print(f"Failed {save_path}: {e}")

def build_pairs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    all_names = get_all_prompt_names(GRPO_DIR)
    
    # 1. Load MDM Runs
    print("Loading MDM runs...")
    load_tasks = [(name, r, GRPO_DIR) for name in all_names for r in range(NUM_RUNS)]
    runs_by_name = {name: {} for name in all_names}
    with ProcessPoolExecutor(max_workers=num_cores) as ex:
        futures = {ex.submit(load_run, task): task for task in load_tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
            task = futures[fut]
            run_idx_ret, data = fut.result()
            if data is not None: runs_by_name[task[0]][run_idx_ret] = data

    # 2. Ranking & Filter
    mdm_pairs = []
    for name in all_names:
        runs = runs_by_name[name]
        if len(runs) < 2: continue
        ranked = sorted(runs.items(), key=lambda kv: kv[1].get("final_metrics", {}).get("Phys_Err", float("inf")))
        best, worst = ranked[0][1], ranked[-1][1]
        if worst.get("final_metrics", {}).get("Phys_Err", 0.0) > best.get("final_metrics", {}).get("Phys_Err", float("inf")):
            mdm_pairs.append((name, best, worst))

    # 3. Load GT
    print("Loading GT...")
    gt_traj_map, gt_metrics_map = {}, {}
    gt_tasks = [(name, GT_DIR) for name, _, _ in mdm_pairs]
    with ProcessPoolExecutor(max_workers=num_cores) as ex:
        futures = {ex.submit(load_gt, task): task for task in gt_tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="GT Loading"):
            res = fut.result()
            if res[1] is not None:
                gt_traj_map[res[0]], gt_metrics_map[res[0]] = res[1], res[2]

    # 4. Save Timestep Pairs
    print("Saving all pairs (MDM and GT)...")
    for name, best_data, worst_data in tqdm(mdm_pairs, desc="Saving"):
        best_err = best_data.get("final_metrics", {}).get("Phys_Err", float("inf"))
        neg_traj = worst_data["x0_trajectory"]
        neg_metrics = worst_data["final_metrics"]

        # Save MDM-best pair
        for t in range(len(best_data["x0_trajectory"])):
            save_to_disk(name, t, best_data["x0_trajectory"], neg_traj, best_data["final_metrics"], neg_metrics, "mdm")

        # Save GT pair (if valid)
        if name in gt_traj_map and gt_metrics_map[name].get("Phys_Err", float("inf")) < best_err:
            gt_traj = gt_traj_map[name]
            gt_met = gt_metrics_map[name]
            for t in range(len(gt_traj)):
                save_to_disk(name, t, gt_traj, neg_traj, gt_met, neg_metrics, "gt")

if __name__ == "__main__":
    build_pairs()