import os
import sys
import numpy as np
from tqdm import tqdm

# Ensure the current directory is in the path so 'utils' can be imported
sys.path.append(os.getcwd())

from utils.compute_physics_metrics import compute_physics_metrics

# ====================================================================== #
#  CONFIGURATION
# ====================================================================== #

DATA_DIR  = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D/new_joint_vecs"
MEAN_PATH = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D/Mean.npy"
STD_PATH  = "/home/npalghat/projects/GenAI/motion-diffusion-model/dataset/HumanML3D/Std.npy"

# Calibrated Thresholds based on Ground Truth analysis
THRESH = {
    "Penetrate":   0.005,  
    "Float":       5.,  # Max GT was 0.203
    "Skate":       2.,  
    "Jitter":      75.0    # Max GT was 54.7
}

def check_gt_quality():
    # 1. Load Normalization Stats
    if not os.path.exists(MEAN_PATH) or not os.path.exists(STD_PATH):
        print(f"Error: Mean or Std files not found at {MEAN_PATH}")
        return
    
    # 2. Collect Files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    files.sort()
    
    if not files:
        print(f"No .npy files found in {DATA_DIR}")
        return

    results_list = []

    # 3. Process Loop
    print(f"\n{'Filename':<15} | {'Transitions':<12} | {'Status'}")
    print("-" * 40)

    for i, fname in enumerate(tqdm(files, desc="Computing physics metrics")):
        try:
            data_raw = np.load(os.path.join(DATA_DIR, fname))
            m = compute_physics_metrics(data_raw)
            results_list.append(m)

            if i < 50 or (i % 500 == 0):
                pen   = m.get('Penetrate', 0)
                flt   = m.get('Float', 0)
                skate = m.get('Skate', 0)
                perr  = m.get('Phys_Err', pen + flt + skate)
                ok    = perr < 50   # mm threshold — tune as needed
                status = "✅" if ok else "❌"
                print(f"{fname:<15} | {pen:>8.3f} | {flt:>9.3f} | {skate:>9.3f} | {perr:>9.3f} | {status}")

        except Exception as e:
            print(f"{fname}: ERROR {e}")
            continue

    # 4. Final Statistics Block
    if not results_list:
        print("No metrics were collected.")
        return

    print("\n" + "="*60)
    print(f"FINAL DATASET STATISTICS ({len(results_list)} files)")
    print("="*60)
    
    metrics_to_stats = ["Penetrate", "Float", "Skate", "SilentSkate", "Jitter", "Transitions"]
    
    print(f"{'Metric':<12} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Median':<10}")
    print("-" * 60)
    
    for key in metrics_to_stats:
        try:
            # Explicitly extract the values into a clean numpy float array
            # This bypasses any object-type issues in the list of dictionaries
            vals = np.array([float(r[key]) for r in results_list if key in r], dtype=np.float64)
            
            if vals.size == 0:
                print(f"{key:<12} | No Data")
                continue

            print(f"{key:<12} | "
                  f"{np.min(vals):<10.4f} | "
                  f"{np.max(vals):<10.4f} | "
                  f"{np.mean(vals):<10.4f} | "
                  f"{np.median(vals):<10.4f}")
        except KeyError:
            print(f"{key:<12} | Key not found in results")
        except Exception as e:
            print(f"{key:<12} | Stat Error: {e}")

if __name__ == "__main__":
    check_gt_quality()