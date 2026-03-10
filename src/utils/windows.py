import logging

import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided

logger = logging.getLogger("cd.utils.windows")


def sample_test_windows(
    x: np.ndarray,
    fail: np.ndarray,
    window: int,
    horizon: int,
    episode_id_offset: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract testing windows according to an array of fail steps.

    Input:
        x: (batch, steps, channels)
        fail: array of ints with failure step (if failure occurs), else -1
        episode_id_offset: added to episode indices in the returned episode_ids

    Samples one window from every trajectory:
    - If the trajectory is a failure, sample the window so that failure occurs
      within horizon steps of the end of the window (if this is not possible, ignore the trajectory)
    - If the trajectory is a success, sample any window

    Returns:
        windows: (n_valid, window, channels)
        y_true: (n_valid,) boolean — True if failure within horizon
        episode_ids: (n_valid,) int — source episode index (+ offset) per window
    """
    assert x.shape[0] == fail.shape[0], 'x and fail are not the same length'

    batch, steps, channels = x.shape
    x_out = np.zeros((batch, window, channels), dtype=x.dtype)
    fail_out = np.zeros((batch,))
    ep_ids = np.zeros((batch,), dtype=np.int64)

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
                    logger.info(
                        f"Skipping traj {i} with fail {fail[i]}: No valid window found.")
                continue
            start = np.random.randint(start_min, start_max + 1)
            f = fail[i] - (start + window)

        x_out[j] = x[i, start:start+window]
        fail_out[j] = f
        ep_ids[j] = episode_id_offset + i
        j += 1

    if verbose:
        logger.info(
            f'Sampled windows from {j} / {len(x)} trajectories w/ fail prop. {(fail_out[:j] > -1).mean()}')

    y_true = fail_out[:j] >= 0
    return x_out[:j], y_true, ep_ids[:j]


def sample_random_windows(
    X: np.ndarray,
    fail: np.ndarray,
    win: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one random window per episode, avoiding failure timesteps.

    For each episode, sample a random window of length ``win`` that ends
    before the failure timestep (if any). Episodes where failure occurs
    too early (fail < win) are discarded.

    Parameters
    ----------
    X : (n_episodes, seq_len, channels)
    fail : (n_episodes,) — -1 for success, >=0 for failure timestep
    win : window length
    rng : numpy random generator

    Returns
    -------
    windows : (n_valid, win, channels)
    valid_mask : (n_episodes,) boolean mask of episodes that yielded a window
    """
    n_episodes, seq_len, _ = X.shape

    max_start = np.where(fail < 0, seq_len - win, fail - win)
    valid_mask = max_start >= 0

    valid_indices = np.where(valid_mask)[0]
    starts = rng.integers(0, max_start[valid_indices] + 1)

    windows = np.array([X[i, s:s + win] for i, s in zip(valid_indices, starts, strict=True)])

    return windows, valid_mask


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


# =============================================================================
# Window Length Estimation
# =============================================================================

def estimate_window(x: np.ndarray, period: int = 1, method: str = 'median') -> int:
    """Estimate window length from the dominant FFT frequency.

    Suited for periodic signals where the fundamental frequency
    determines a natural window size.

    x: (n_episodes, seq_len, n_features)
    """
    steps = x.shape[1]
    fft_magnitudes = np.abs(np.fft.rfft(x, axis=1))
    fft_magnitudes[:, 0, :] = 0

    peak_indices = np.argmax(fft_magnitudes, axis=1)
    peak_indices[peak_indices == 0] = 1

    frequencies = peak_indices / steps

    avg_frequency = np.mean(frequencies) if method == 'mean' else np.median(frequencies)

    estimated_window = period / avg_frequency
    window = int(round(estimated_window))
    logger.debug(f"periods={period}, freq={avg_frequency:.4f} ({method}) → window={window}")

    return window


def estimate_window_acf(x: np.ndarray, min_window: int = 10) -> int:
    """Estimate window as decorrelation time via ACF zero-crossing.

    Computes the autocorrelation for each episode/feature, averages across
    all of them, and returns the lag of the first zero-crossing.
    Suited for non-periodic signals.

    x: (n_episodes, seq_len, n_features)
    """
    n_episodes, seq_len, n_features = x.shape
    x_centered = x - x.mean(axis=1, keepdims=True)

    n_fft = 2 * seq_len
    X_freq = np.fft.rfft(x_centered, n=n_fft, axis=1)
    acf_full = np.fft.irfft(X_freq * np.conj(X_freq), n=n_fft, axis=1)
    acf_full = acf_full[:, :seq_len, :]
    acf_full = acf_full / (acf_full[:, 0:1, :] + 1e-12)

    mean_acf = acf_full.mean(axis=(0, 2))

    zero_crossings = np.where(mean_acf[1:] <= 0)[0]
    if len(zero_crossings) > 0:
        window = int(zero_crossings[0]) + 1
    else:
        window = seq_len // 2

    window = max(min_window, window)
    logger.debug(f"ACF window estimate: {window}")
    return window


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
