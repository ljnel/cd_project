import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Tuple
import torch


def sample_test_windows(x: np.ndarray, fail: np.ndarray, window: int, horizon: int, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract testing windows according to an array of fail steps.

    Input:
        x: (batch, steps, channels)
        fail: array of ints with failure step (if failure occurs), else -1

    Samples one window from every trajectory:
    - If the trajectory is a failure, sample the window so that failure occurs
      within horizon steps of the end of the window (if this is not possible, ignore the trajectory)
    - If the trajectory is a success, sample any window

    Also, return the fail array with the end of the window subtracted, whenever failure occurs.
    """
    assert x.shape[0] == fail.shape[0], 'x and fail are not the same length'

    batch, steps, channels = x.shape
    x_out = np.zeros((batch, window, channels), dtype=x.dtype)
    fail_out = np.zeros((batch,))

    j = 0  # num of valid trajs found so far
    for i in range(batch):
        if fail[i] == -1:
            max_start = steps - window
            if max_start < 0:
                continue  # traj too short

            start = np.random.randint(0, max_start + 1)
            f = -1
        else:
            end_min = fail[i] - horizon
            end_max = fail[i]

            start_min = end_min - window
            start_max = end_max - window

            start_min = max(start_min, 0)  # don't let start_min < 0
            # don't let start_max > steps - window
            start_max = min(start_max, steps - window)

            if start_min > start_max:  # includes the case start_max < 0
                if verbose:
                    print(
                        f"Skipping traj {i} with fail {fail[i]}: No valid window found.")
                continue
            start = np.random.randint(start_min, start_max + 1)
            f = fail[i] - (start + window)

        x_out[j] = x[i, start:start+window]
        fail_out[j] = f
        j += 1

    if verbose:
        print(
            f'Sampled windows from {j} / {len(x)} trajectories w/ fail prop. {(fail_out[:j] > -1).mean()}')

    return x_out[:j], fail_out[:j]


def strided_window_view(ds, window: int, stride: int):
    """
    Create a strided view of windows from a dataset.

    ds: numpy array of shape (n_episodes, n_steps, n_chan)

    Returns: view of shape (n_episodes, n_windows, window, n_chan)
    """
    ds = np.asarray(ds, dtype=np.float32)
    n_episodes, n_steps, n_chan = ds.shape
    assert window <= n_steps

    n_windows = (n_steps - window) // stride + 1

    new_shape = (n_episodes, n_windows, window, n_chan)
    new_strides = (
        ds.strides[0],
        ds.strides[1] * stride,
        ds.strides[1],
        ds.strides[2],
    )
    return as_strided(ds, shape=new_shape, strides=new_strides)


class WindowDataset(torch.utils.data.Dataset):
    """
    A class for getting windows from a dataset for training torch models.    
    """

    def __init__(self, ds: np.ndarray, window: int, stride):
        """
        ds: numpy array of shape (n_episodes, n_steps, n_chan)
        """
        self.ds = torch.tensor(ds, dtype=torch.float32)
        self.window = window
        self.stride = stride
        self.n_episodes, self.n_steps, self.n_chan = self.ds.shape
        assert self.window <= self.n_steps

        self.windows_per_ep = (self.n_steps - self.window) // self.stride + 1

    def __len__(self):
        return self.n_episodes * self.windows_per_ep

    def __getitem__(self, idx):
        ep_idx = idx // self.windows_per_ep
        win_idx = idx % self.windows_per_ep
        step_idx = win_idx * self.stride
        window = self.ds[ep_idx, step_idx:step_idx + self.window]
        return window
