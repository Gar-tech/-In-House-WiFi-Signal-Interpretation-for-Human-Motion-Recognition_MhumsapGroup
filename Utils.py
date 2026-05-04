import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CSIDataset(Dataset):
    def __init__(self, csv_file, amp_folder, max_time_steps=1000):
        self.df = pd.read_csv(csv_file)
        self.activity_col = "user_1_activity"
        self.df = self.df.dropna(subset=[self.activity_col]).reset_index(drop=True)
        self.amp_folder = amp_folder
        self.max_time = max_time_steps
        self.label_map = {
            "nothing": 0, "walk": 1, "rotation": 2, "jump": 3, "wave": 4,
            "lie_down": 5, "pick_up": 6, "sit_down": 7, "stand_up": 8
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row["label"]).strip()
        label = self.label_map.get(str(row[self.activity_col]).strip(), 0)

        amp_path = os.path.join(self.amp_folder, file_id + ".npy")
        if not os.path.exists(amp_path):
            return torch.zeros((30, self.max_time)), label

        amp = np.load(amp_path)
        amp = np.squeeze(amp)
        if amp.ndim > 2: amp = amp.reshape(amp.shape[0], -1)

        # Difference CSI: Subtract previous time step to highlight movement
        diff_amp = np.diff(amp, axis=0, prepend=amp[:1, :])
        
        # Z-Score Normalization on the delta
        amp = (diff_amp - np.mean(diff_amp)) / (np.std(diff_amp) + 1e-6)

        # Feature/Time padding
        if amp.shape[1] > 30: amp = amp[:, :30]
        elif amp.shape[1] < 30: amp = np.pad(amp, ((0,0),(0,30-amp.shape[1])), 'constant')
        
        if amp.shape[0] > self.max_time: amp = amp[:self.max_time, :]
        elif amp.shape[0] < self.max_time: amp = np.pad(amp, ((0, self.max_time - amp.shape[0]), (0, 0)), 'constant')

        return torch.tensor(amp, dtype=torch.float32).permute(1, 0), torch.tensor(label, dtype=torch.long)