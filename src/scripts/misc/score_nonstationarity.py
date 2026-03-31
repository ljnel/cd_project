#!/usr/bin/env python3
"""Diagnostic plots for score non-stationarity."""

import matplotlib.pyplot as plt
import numpy as np

from config.detectors import get_detector
from data.datasets import load_episodes, normalize_channels
from utils.plotting import FULL_WIDTH, setup_style
from utils.windows import strided_window_view

setup_style()


def score_episodes_by_timestep(detector, episodes, stride=5):
    win = detector.window
    all_scores = []
    for ep in episodes:
        windows = strided_window_view(ep[np.newaxis], win, stride=stride)[0]
        scores = detector.score_samples(windows)
        all_scores.append(scores)
    n_win = (episodes.shape[1] - win) // stride + 1
    timesteps = np.arange(n_win) * stride + (win - 1)
    return np.array(all_scores), timesteps


envs = ["hopper", "half_cheetah"]
fig, axes = plt.subplots(4, len(envs), figsize=(FULL_WIDTH, 7), sharex="col")

for col, env in enumerate(envs):
    x_success, _ = load_episodes(env, dataset='train')
    x_success = x_success[:400]
    scaler, x_success = normalize_channels(x_success)

    x_det_train = x_success[:200]
    x_norm_cal = x_success[200:400]

    detector = get_detector("basis", env=env)
    detector.cal_fraction = 0.0
    detector.fit(x_det_train)

    scores, timesteps = score_episodes_by_timestep(detector, x_norm_cal)

    mean_t = scores.mean(axis=0)
    median_t = np.median(scores, axis=0)
    std_t = scores.std(axis=0)
    q25 = np.percentile(scores, 25, axis=0)
    q75 = np.percentile(scores, 75, axis=0)
    iqr_t = q75 - q25
    cv_t = std_t / np.maximum(mean_t, 1e-8)

    # Row 0: mean and median
    ax = axes[0, col]
    ax.plot(timesteps, mean_t, label="mean", linewidth=0.8)
    ax.plot(timesteps, median_t, label="median", linewidth=0.8)
    ax.set_ylabel("Level")
    ax.set_title(f"{env.replace('_', ' ').title()}", fontsize=9)
    ax.legend(fontsize=6)

    # Row 1: std and IQR
    ax = axes[1, col]
    ax.plot(timesteps, std_t, label="std", linewidth=0.8)
    ax.plot(timesteps, iqr_t, label="IQR", linewidth=0.8)
    ax.set_ylabel("Spread")
    ax.legend(fontsize=6)

    # Row 2: CV
    ax = axes[2, col]
    ax.plot(timesteps, cv_t, linewidth=0.8, color="C2")
    ax.set_ylabel("CV (std/mean)")

    # Row 3: which timestep achieves the max z-score per trajectory
    z_scores = (scores - mean_t) / np.maximum(std_t, 1e-8)
    max_t_idx = np.argmax(z_scores, axis=1)
    ax = axes[3, col]
    ax.hist(timesteps[max_t_idx], bins=50, color="C3", alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_xlabel("Timestep")

axes[3, 0].set_title("Timestep of max z-score", fontsize=7, loc="left")

fig.tight_layout()
plt.savefig("score_nonstationarity.png", dpi=150)
plt.show()
print("Done.")
