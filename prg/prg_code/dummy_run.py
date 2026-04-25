import torch
import numpy as np
from reward_model.model import create_reward_model
from pathlib import Path
import pickle

# 1. Get the directory where dummy_run.py is located
current_dir = Path(__file__).parent

# 2. Navigate up one level to project root, then down into the data folder
# file_path = current_dir.parent / "sample_motion" / "new_joints_processed" / "000000.npy"
file_path = "dummy_data/transfer_subsets/positive/000002.pkl"

# 1. Open and load the pickle file
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"--- Successfully loaded: {file_path} ---\n")

    # 2. Check the type of the root object
    print(f"Data Type: {type(data)}")

    # 3. If it's a dictionary, check the keys
    if isinstance(data, dict):
        print(f"Keys found: {list(data.keys())}")
        for key, value in data.items():
            # Check for numpy arrays or torch tensors specifically
            if hasattr(value, 'shape'):
                print(f"  - {key}: {type(value)} with shape {value.shape}")
            else:
                print(f"  - {key}: {type(value)}")

    # 4. If it's a list or tuple, check the length
    elif isinstance(data, (list, tuple)):
        print(f"Length of sequence: {len(data)}")
        if len(data) > 0:
            print(f"Type of first element: {type(data[0])}")

    # 5. Quick peek at the actual data
    print("\n--- Data Preview ---")
    print(data)

except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
# Dummy inputs — no dataset needed
# B, T, D = 2, 64, 263          # batch=2, 64 frames, 263 HumanML3D features

# B = len(motion)
# T = motion.shape[1]
# D = motion.shape[2]

# Fake mean/std (substitute for data/mean.npy and data/std.npy)
# mean = np.zeros(D, dtype=np.float32)
# std  = np.ones(D,  dtype=np.float32)

# Build model (random weights — no checkpoint needed)
# model = create_reward_model()
# model.eval()

# # Normalize and run
# x = torch.tensor((motion - mean) / (std + 1e-8))  # [B, T, 263]
# t = torch.randint(0, 1000, (B,))                   # random timestep

# with torch.no_grad():
#     reward = model(x, t)

# print("reward shape:", reward.shape)   # expect: torch.Size([2])
# print("reward values:", reward)