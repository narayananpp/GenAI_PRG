import os
import pickle
import numpy as np
import torch as th
from torch.utils.data import Dataset

class RewardPairedDataset(Dataset):
    """
    Loads pre-built reward pairs from the consolidated reward_samples directory.
    Each .pkl file contains: 
    {
        "name": str, "t": int,
        "positive_frame": (T, 263), 
        "negative_frame": (T, 263),
        "positive_metrics": dict,
        "negative_metrics": dict,
        "positive_source": str
    }
    """
    def __init__(
        self,
        data_dir,
        mean,
        std,
        max_len=196,
        split="train",
        val_fraction=0.1,
        seed=42,
    ):
        self.mean = mean
        self.std = std + 1e-8
        self.max_len = max_len

        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        if not all_files:
            raise ValueError(f"No .pkl files found in {data_dir}")

        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        
        n_val = max(1, int(len(all_files) * val_fraction))
        if split == "val":
            selected_indices = indices[:n_val]
        else:
            selected_indices = indices[n_val:]
            
        self.file_paths = [os.path.join(data_dir, all_files[i]) for i in selected_indices]
        print(f"[Dataset] split={split}: Found {len(self.file_paths)} samples.")

    def __len__(self):
        return len(self.file_paths)

    def _process_motion(self, motion):
        T = motion.shape[0]
        motion = (motion - self.mean) / self.std
        if T >= self.max_len:
            return motion[:self.max_len], self.max_len
        padded = np.zeros((self.max_len, motion.shape[1]), dtype=np.float32)
        padded[:T] = motion
        return padded, T

    def __getitem__(self, idx):
        with open(self.file_paths[idx], "rb") as f:
            data = pickle.load(f)

        pos_motion, pos_len = self._process_motion(data["positive_frame"])
        neg_motion, neg_len = self._process_motion(data["negative_frame"])
        
        # Retrieve the timestep baked into the file
        t = data.get("t", 0)

        return (
            th.from_numpy(pos_motion).float(), 
            th.from_numpy(neg_motion).float(), 
            th.tensor(pos_len, dtype=th.long),
            th.tensor(neg_len, dtype=th.long),
            th.tensor(int(t), dtype=th.long) # Added timestep t
        )

def load_motion_data(data_dir, mean_path, std_path, **kwargs):
    mean = np.load(mean_path) if isinstance(mean_path, str) else mean_path
    std  = np.load(std_path) if isinstance(std_path, str) else std_path
    
    if hasattr(mean, 'shape') and mean.shape[0] != 263: mean = mean.squeeze()
    if hasattr(std, 'shape') and std.shape[0] != 263: std = std.squeeze()

    return RewardPairedDataset(data_dir, mean, std, **kwargs)

def padding_mask(lengths, max_len):
    return th.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)