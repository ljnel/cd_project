"""Heatmap of the (T*D) x (T*D) and Tucker (K*M) x (K*M) covariance matrices
over success windows."""

import argparse
import sys

sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import numpy as np

from algs.tucker import Tucker13
from config.envs import ENV_INFO
from config.tasks import TASK_CONFIGS
from tasks.safety_monitor import SafetyMonitor
from utils.windows import strided_window_view

parser = argparse.ArgumentParser(description="Covariance heatmap for any environment.")
parser.add_argument("--env", type=str, required=True,
                    help=f"Environment name. Available: {list(TASK_CONFIGS.keys())}")
args = parser.parse_args()

env = args.env
if env not in TASK_CONFIGS:
    raise ValueError(f"No task config for '{env}'. Available: {list(TASK_CONFIGS.keys())}")

task_cfg = TASK_CONFIGS[env]
env_info = ENV_INFO.get(env)
display_name = env_info.display_name if env_info else env

# Load data via SafetyMonitor (handles loading, splitting, normalization, windowing)
task = SafetyMonitor(task_cfg)
x_train, x_test = task.get_train_test()
y_true = task.y_true

window = task_cfg.win
D = x_train.shape[-1]

# Dense training windows from full success episodes
stride = window // 2
max_train_windows = 1000
wins_train = strided_window_view(x_train[:, window:, :], window=window, stride=stride)
wins_train = wins_train.reshape(-1, window, D)
rng = np.random.default_rng(0)
if len(wins_train) > max_train_windows:
    wins_train = wins_train[rng.choice(len(wins_train), max_train_windows, replace=False)]

# Test windows from SafetyMonitor (already (n, win, D))
wins_succ = x_test[~y_true]
wins_fail = x_test[y_true]
print(f"Train windows: {len(wins_train)}, Plot (success): {len(wins_succ)}, "
      f"Plot (failure): {len(wins_fail)}")

# --- Raw covariance (skip if T*D is too large) ---
RAW_COV_MAX_DIM = 5000
raw_dim = window * D
show_raw = raw_dim <= RAW_COV_MAX_DIM
if show_raw:
    flat_plot = wins_succ.reshape(len(wins_succ), -1)
    print(f"Raw feature dim: {flat_plot.shape[1]}")
    cov_raw = np.cov(flat_plot, rowvar=False)
else:
    print(f"Skipping raw covariance: T*D={raw_dim} exceeds {RAW_COV_MAX_DIM}")

# --- Tucker covariance (success + failure) ---
K, M = 10, 15
tucker = Tucker13(n_spatial=M, n_temporal=K).fit(wins_train)
cores_succ = tucker.transform(wins_succ)
cores_fail = tucker.transform(wins_fail)
print(f"Tucker feature dim: {cores_succ.shape[1]} (K={K}, M={M})")
cov_tucker_succ = np.cov(cores_succ, rowvar=False)
cov_tucker_fail = np.cov(cores_fail, rowvar=False)

# --- Plot ---
n_plots = 3 if show_raw else 2
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

ax_idx = 0

# Raw (success)
if show_raw:
    vmax_raw = np.percentile(np.abs(cov_raw), 99)
    im0 = axes[ax_idx].imshow(cov_raw, cmap="RdBu_r", vmin=-vmax_raw, vmax=vmax_raw, aspect="equal")
    for i in range(1, D):
        pos = i * window
        axes[ax_idx].axhline(pos - 0.5, color="k", lw=0.3, alpha=0.5)
        axes[ax_idx].axvline(pos - 0.5, color="k", lw=0.3, alpha=0.5)
    axes[ax_idx].set_xlabel("Feature index (T×D)")
    axes[ax_idx].set_ylabel("Feature index (T×D)")
    axes[ax_idx].set_title(f"Raw success (T={window}, D={D}, dim={window*D})")
    fig.colorbar(im0, ax=axes[ax_idx], shrink=0.8)
    ax_idx += 1

# Tucker — shared color scale for both
vmax_tk = max(np.percentile(np.abs(cov_tucker_succ), 99),
              np.percentile(np.abs(cov_tucker_fail), 99))

# Tucker success
im1 = axes[ax_idx].imshow(cov_tucker_succ, cmap="RdBu_r", vmin=-vmax_tk, vmax=vmax_tk, aspect="equal")
for i in range(1, M):
    pos = i * K
    axes[ax_idx].axhline(pos - 0.5, color="k", lw=0.3, alpha=0.5)
    axes[ax_idx].axvline(pos - 0.5, color="k", lw=0.3, alpha=0.5)
axes[ax_idx].set_xlabel("Feature index (K×M)")
axes[ax_idx].set_ylabel("Feature index (K×M)")
axes[ax_idx].set_title(f"Tucker success (K={K}, M={M})")
fig.colorbar(im1, ax=axes[ax_idx], shrink=0.8)
ax_idx += 1

# Tucker failure
im2 = axes[ax_idx].imshow(cov_tucker_fail, cmap="RdBu_r", vmin=-vmax_tk, vmax=vmax_tk, aspect="equal")
for i in range(1, M):
    pos = i * K
    axes[ax_idx].axhline(pos - 0.5, color="k", lw=0.3, alpha=0.5)
    axes[ax_idx].axvline(pos - 0.5, color="k", lw=0.3, alpha=0.5)
axes[ax_idx].set_xlabel("Feature index (K×M)")
axes[ax_idx].set_ylabel("Feature index (K×M)")
axes[ax_idx].set_title(f"Tucker failure (K={K}, M={M})")
fig.colorbar(im2, ax=axes[ax_idx], shrink=0.8)

fig.suptitle(f"Covariance of windows — {display_name}", fontsize=14)
fig.tight_layout()
fig.savefig(f"results/cov_heatmap_{env}.pdf", dpi=150)
print(f"Saved to results/cov_heatmap_{env}.pdf")
plt.show()
