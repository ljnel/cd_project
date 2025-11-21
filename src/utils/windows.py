import numpy as np
from typing import Tuple


def make_windows(x: np.ndarray, fail: np.ndarray, window: int, horizon: int, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        x: (batch, steps, channels)
        fail: array of ints with failure step (if failure occurs), else -1

    For every trajectory, sample a window of length window.
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
                 continue # traj too short
            
            start = np.random.randint(0, max_start + 1)
            f = -1
        else:
            end_min = fail[i] - horizon
            end_max = fail[i]

            start_min = end_min - window
            start_max = end_max - window

            start_min = max(start_min, 0)  # don't let start_min < 0
            start_max = min(start_max, steps - window)  # don't let start_max > steps - window

            if start_min > start_max:  # includes the case start_max < 0
                if verbose:
                    print(f"Skipping traj {i} with fail {fail[i]}: No valid window found.")
                continue
            start = np.random.randint(start_min, start_max + 1)
            f = fail[i] - (start + window)
    
        x_out[j] = x[i, start:start+window]
        fail_out[j] = f
        j += 1

    if verbose:
        print(f'Sampled windows from {j} / {len(x)} trajectories')

    return x_out[:j], fail_out[:j]