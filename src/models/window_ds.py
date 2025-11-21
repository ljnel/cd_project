import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class WindowDataset(Dataset):
    def __init__(self, ds, fails, past_win: int, fut_win: int, stride: int = 1):
        """
        ds: numpy array of shape (n_episodes, n_steps, n_chan)
        fails: (n_episodes,) with 0 for succ and step > 0 for fail at step
        """
        self.ds = torch.tensor(ds, dtype=torch.float32)
        self.past_win = past_win
        self.fut_win = fut_win
        self.stride = stride
        self.n_episodes, self.n_steps, self.n_chan = self.ds.shape
        self.window = self.past_win + self.fut_win
        assert self.window <= self.n_steps

        self.windows_per_ep = (self.n_steps - self.window) // self.stride + 1

        self.fails = np.where(fails == 0, np.full_like(fails, 1000), fails)

    def __len__(self):
        # total number of valid windows across all episodes
        return self.n_episodes * self.windows_per_ep

    def __getitem__(self, idx):
        ep_idx = idx // self.windows_per_ep
        win_idx = idx % self.windows_per_ep
        step_idx = win_idx * self.stride        
        window = self.ds[ep_idx, step_idx:step_idx + self.past_win]
        will_fail = self.fails[ep_idx] < step_idx + self.window  # has failure occurred by the end of the horizon?
        return window, will_fail