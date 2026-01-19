import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_contours(f, ax, **plot_kwargs):
    "Make a contour plot of a vectorized function on the given axes."

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(*xlim, 100)  # type: ignore
    y = np.linspace(*ylim, 100)  # type: ignore
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = f(points)
    Z = vals.reshape(X.shape)

    contours = ax.contour(X, Y, Z, alpha=0.9, **plot_kwargs)
    ax.clabel(contours)
    return contours


def plot_level_set(f, alpha, ax):
    "Plot the alpha-level set of a vectorized function on the given axes."

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(*xlim, 100)  # type: ignore
    y = np.linspace(*ylim, 100)  # type: ignore
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = f(points)
    Z = vals.reshape(X.shape)

    ax.contour(X, Y, Z, levels=[0.0, alpha], cmap="Blues", alpha=0.9)


def plot_func(f, ax, **plot_kwargs):
    "Plot a vectorized function on [-1, 1]."
    ts = np.linspace(-1, 1, 100)
    ax.plot(ts, f(ts), **plot_kwargs)


def plot_kern_mat(K, y, ax, title=None):
    "Plot a kernel matrix, ordered by label."
    idx = np.argsort(y)  # reorder by label
    K_sorted = K[np.ix_(idx, idx)]

    sns.heatmap(K_sorted, cmap='viridis', ax=ax)
    if title is not None:
        ax.set_title(title)


def plot_channels(x, y, max_channels=9, max_batches=50):
    import math
    n_batch, n_step, n_channel = x.shape

    if n_channel > max_channels:
        channel_indices = np.random.choice(
            n_channel, max_channels, replace=False)
        channel_indices.sort()
        print(f"Sampling {max_channels} random channels out of {n_channel}...")
    else:
        channel_indices = np.arange(n_channel)

    if n_batch > max_batches:
        batch_indices = np.random.choice(n_batch, max_batches, replace=False)
        batch_indices.sort()
        print(f"Sampling {max_batches} random batches out of {n_batch}...")
    else:
        batch_indices = np.arange(n_batch)

    num_plots = len(channel_indices)
    cols = int(math.ceil(math.sqrt(num_plots)))
    rows = int(math.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(
        4 * cols, 3 * rows), constrained_layout=True)

    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, ch_idx in enumerate(channel_indices):
        ax = axes[i]

        for b_idx in batch_indices:
            color = 'red' if y[b_idx] else 'green'
            alpha = 0.6 if n_batch > 10 else 1.0  # Transparency for overlapping lines

            ax.plot(x[b_idx, :, ch_idx], color=color, linewidth=1, alpha=alpha)

        ax.set_title(f"Channel {ch_idx}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    return fig


def save_plot(
    name: str,
    ax=None,
    *,
    subfolder: str = "outputs/plots",
    ext: str = "png",
    dpi: int = 300,
    bbox_inches: str | None = "tight",
) -> Path:
    caller_stem = Path(inspect.stack()[1].filename).stem
    base_stem = f"{caller_stem}_{name}"

    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    outfile = output_dir / f"{base_stem}.{ext}"

    fig = ax.figure if ax is not None else plt.gcf()
    fig.savefig(outfile, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved to: {outfile}")
    return outfile


def plot_map(f, ax, **plot_kwargs):
    "Make a contour plot of a vectorized function on the given axes."

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(*xlim, 100)  # type: ignore
    y = np.linspace(*ylim, 100)  # type: ignore
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = f(points)
    Z = vals.reshape(X.shape)

    im = ax.pcolormesh(X, Y, Z, shading="auto", **plot_kwargs)
    plt.colorbar(im, ax=ax)
    return vals
