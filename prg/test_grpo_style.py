import os
import pickle
import numpy as np

# Configuration - update these to match your paths
GRPO_DIR = "/data3/npalghat/reward_dataset/MDM_preds_GRPO"
NUM_RUNS = 5

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

def check_physics_differences():
    all_names = get_all_prompt_names(GRPO_DIR)
    differences = []
    
    print(f"{'Prompt':<30} | {'Best Err':<10} | {'Worst Err':<10} | {'Diff':<10}")
    print("-" * 70)
    
    for name in all_names:
        run_errors = []
        for r in range(NUM_RUNS):
            # Try to load either _runX or the default file
            path = os.path.join(GRPO_DIR, f"{name}_run{r}.pkl")
            if not os.path.exists(path) and r == 0:
                path = os.path.join(GRPO_DIR, f"{name}.pkl")
            
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    err = data.get("final_metrics", {}).get("Phys_Err", None)
                    if err is not None:
                        run_errors.append(err)
        
        if len(run_errors) >= 2:
            best = min(run_errors)
            worst = max(run_errors)
            diff = worst - best
            differences.append(diff)
            print(f"{name[:28]:<30} | {best:<10.2f} | {worst:<10.2f} | {diff:<10.2f}")

    if differences:
        print("-" * 70)
        print(f"Average Physics Error Difference: {np.mean(differences):.2f}")
        print(f"Median Physics Error Difference: {np.median(differences):.2f}")
    else:
        print("No valid pairs found.")

if __name__ == "__main__":
    check_physics_differences()