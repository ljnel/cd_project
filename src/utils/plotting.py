import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger("cd.utils.plotting")

# IEEE two-column figure widths (inches)
COL_WIDTH = 3.5
FULL_WIDTH = 7.16

# Colorblind-safe survival/failure colors (Tol bright)
SURVIVAL_COLOR = "#228833"  # green
FAILURE_COLOR = "#EE6677"  # red/pink


def setup_style():
    """Configure matplotlib for publication (IEEE two-column, serif fonts)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "text.usetex": True,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


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


def plot_level_set(f, alpha, ax=None):
    "Plot the alpha-level set of a vectorized function on the given axes."
    if ax is None:
        ax = plt.gca()

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


def plot_gram(K: np.ndarray, y: np.ndarray, ax: plt.Axes | None = None, title: str | None = None):
    "Plot a kernel matrix, ordered by label."
    assert np.isin(y, [0, 1]).all(), f"y must be binary, got unique values {np.unique(y)}"
    if ax is None:
        ax = plt.gca()
    idx = np.argsort(y)  # reorder by label
    K_sorted = K[np.ix_(idx, idx)]

    sns.heatmap(K_sorted, cmap='viridis', ax=ax)
    if title is not None:
        ax.set_title(title)


def plot_channels(x: np.ndarray, y: np.ndarray, max_channels: int = 9, max_batches: int = 20):
    """Plot per-channel time series colored by binary label.

    Parameters
    ----------
    x : ndarray of shape (n_batch, n_steps, n_channels)
    y : ndarray of shape (n_batch,)
        Binary label: 0 = green, 1 = red.
    max_channels : int
        Maximum number of channels to plot (randomly sampled if exceeded).
    max_batches : int
        Maximum number of trajectories per subplot (randomly sampled if exceeded).
    """
    import math
    n_batch, n_step, n_channel = x.shape

    if n_channel > max_channels:
        channel_indices = np.random.choice(
            n_channel, max_channels, replace=False)
        channel_indices.sort()
        logger.debug(f"Sampling {max_channels} random channels out of {n_channel}...")
    else:
        channel_indices = np.arange(n_channel)

    if n_batch > max_batches:
        batch_indices = np.random.choice(n_batch, max_batches, replace=False)
        batch_indices.sort()
        logger.debug(f"Sampling {max_batches} random batches out of {n_batch}...")
    else:
        batch_indices = np.arange(n_batch)

    num_plots = len(channel_indices)
    cols = int(math.ceil(math.sqrt(num_plots)))
    rows = int(math.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(
        4 * cols, 3 * rows), constrained_layout=True)

    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, ch_idx in enumerate(channel_indices):
        ax = axes[i]

        for b_idx in batch_indices:
            color = FAILURE_COLOR if y[b_idx] else SURVIVAL_COLOR
            alpha = 0.6 if n_batch > 10 else 1.0  # Transparency for overlapping lines

            ax.plot(x[b_idx, :, ch_idx], color=color, linewidth=1, alpha=alpha)

        ax.set_title(f"Channel {ch_idx}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    return fig


def plot_heatmaps(
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int = 4,
    cmap: str = "viridis",
):
    """Heatmap grid of sampled trajectories, split by binary label.

    Parameters
    ----------
    x : ndarray of shape (N, T, D)
    y : ndarray of shape (N,), binary (0 or 1).
    n_samples : int
        Number of trajectories to sample per class.
    cmap : str
        Colormap for the heatmaps.
    """
    assert x.ndim == 3
    assert y.ndim == 1 and len(y) == len(x)
    assert np.isin(y, [0, 1]).all(), f"y must be binary, got unique values {np.unique(y)}"

    idx_false = np.where(y == 0)[0]
    idx_true = np.where(y == 1)[0]

    rng = np.random.default_rng()
    pick_f = rng.choice(idx_false, min(n_samples, len(idx_false)), replace=False)
    pick_t = rng.choice(idx_true, min(n_samples, len(idx_true)), replace=False)

    cols = max(len(pick_f), len(pick_t))
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 5), constrained_layout=True)
    if cols == 1:
        axes = axes[:, None]

    for j in range(cols):
        # Top row: label=0 (survived)
        ax = axes[0, j]
        if j < len(pick_f):
            ax.imshow(x[pick_f[j]].T, aspect="auto", cmap=cmap, interpolation="nearest")
            ax.set_ylabel("Dim") if j == 0 else None
            ax.set_xlabel("Step")
        else:
            ax.axis("off")

        # Bottom row: label=1 (failed)
        ax = axes[1, j]
        if j < len(pick_t):
            ax.imshow(x[pick_t[j]].T, aspect="auto", cmap=cmap, interpolation="nearest")
            ax.set_ylabel("Dim") if j == 0 else None
            ax.set_xlabel("Step")
        else:
            ax.axis("off")

    axes[0, 0].set_title("Survived (y=0)", loc="left", fontsize=10)
    axes[1, 0].set_title("Failed (y=1)", loc="left", fontsize=10)

    return fig


def plot_hists(
    x: np.ndarray,
    y: np.ndarray,
    n_dims: int = 4,
    cmap: str = "viridis",
    n_trajs: int = 6,
    most_different: bool = False,
):
    """Density comparison of sampled dimensions, split by binary label.

    For each selected observation dimension, overlays per-trajectory KDE plots
    to show distributional stability. Top row = survived (y=0),
    bottom = failed (y=1).

    Parameters
    ----------
    x : ndarray of shape (N, T, D)
    y : ndarray of shape (N,), binary (0 or 1).
    n_dims : int
        Number of observation dimensions to sample.
    cmap : str
        Unused, kept for signature compatibility with plot_heatmaps.
    n_trajs : int
        Number of trajectories to overlay per class per dimension.
    most_different : bool
        If True, pick dimensions with largest KS distance between classes.
        If False, pick randomly.
    """
    from scipy.stats import ks_2samp

    assert x.ndim == 3
    assert y.ndim == 1 and len(y) == len(x)
    assert np.isin(y, [0, 1]).all(), f"y must be binary, got unique values {np.unique(y)}"

    N, T, D = x.shape
    rng = np.random.default_rng()
    n_dims = min(n_dims, D)

    if most_different:
        ks_max_trajs = 50
        x_succ = x[y == 0][:ks_max_trajs]
        x_fail = x[y == 1][:ks_max_trajs]
        ks_scores = [
            ks_2samp(x_succ[:, :, d].ravel(), x_fail[:, :, d].ravel()).statistic
            for d in range(D)
        ]
        dim_idx = np.argsort(ks_scores)[-n_dims:]
        dim_idx.sort()
    else:
        dim_idx = np.sort(rng.choice(D, n_dims, replace=False))

    idx_succ = np.where(y == 0)[0]
    idx_fail = np.where(y == 1)[0]
    pick_succ = rng.choice(idx_succ, min(n_trajs, len(idx_succ)), replace=False)
    pick_fail = rng.choice(idx_fail, min(n_trajs, len(idx_fail)), replace=False)

    fig, axes = plt.subplots(2, n_dims, figsize=(3 * n_dims, 5), constrained_layout=True)
    if n_dims == 1:
        axes = axes[:, None]

    for j, d in enumerate(dim_idx):
        ax = axes[0, j]
        for i in pick_succ:
            sns.kdeplot(x[i, :, d], ax=ax, color=SURVIVAL_COLOR, alpha=0.5)
        ax.set_title(f"Dim {d}")
        if j == 0:
            ax.set_ylabel("Density")

        ax = axes[1, j]
        for i in pick_fail:
            sns.kdeplot(x[i, :, d], ax=ax, color=FAILURE_COLOR, alpha=0.5)
        ax.set_xlabel("Value")
        if j == 0:
            ax.set_ylabel("Density")

    axes[0, 0].set_title(f"Success (y=0) — Dim {dim_idx[0]}", loc="left", fontsize=10)
    axes[1, 0].set_title("Failure (y=1)", loc="left", fontsize=10)

    return fig


def plot_detection_curves(leads_by_label, *, ax=None, lead_max=None, min_lead=0,
                          as_percent=True, event_name="event"):
    """Cumulative fraction of events caught vs lead time, one curve per label.

    Parameters
    ----------
    leads_by_label : dict[str, ndarray]
        Label -> `(N,)` signed lead-time array (NaN = missed), as returned by
        `eval.survival.detection_lead_times`.
    ax : matplotlib Axes, optional
        Axes to draw on; a new square figure is created if omitted.
    lead_max : int, optional
        Largest lead time on the x-axis; defaults to the max lead across labels.
    min_lead : int
        Smallest lead time on the x-axis. The default 0 stops the curve at the
        event step; a negative value extends it past the event to show detections
        made *after the fact*, which plateau at the total detection rate. A dotted
        marker is drawn at the event (lead 0) when the curve extends past it.
    as_percent : bool
        Plot the y-axis (and legend EDR) as a percentage rather than a fraction.
    event_name : str
        Noun for the detected event used in the axis labels (e.g. "collision",
        "failure"); pluralized with a trailing "s".

    Each curve rises (as the lead shrinks) from ~0 toward the total detection
    rate, with the value at lead 1 the Early Detection Rate shown in the legend
    and lead 0 the at-impact rate. Misses (NaN) are excluded. The x-axis is
    inverted so the event sits on the right. Returns the Axes (save via `save_plot`).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))
    if lead_max is None:
        lead_max = int(max(np.nanmax(v) for v in leads_by_label.values()))
    grid = np.arange(min_lead, lead_max + 1)
    pct = r"\%" if plt.rcParams.get("text.usetex", False) else "%"
    scale = 100 if as_percent else 1
    for label, leads in leads_by_label.items():
        incidence = (leads[:, None] >= grid).mean(axis=0)   # fraction caught with lead >= L
        edr = (leads >= 1).mean()                           # EDR: strictly before the event
        ax.plot(grid, scale * incidence, label=f"{label} (EDR={scale * edr:.0f}{pct})")
    if min_lead < 0:
        ax.axvline(0, color="0.6", lw=0.8, ls=":")          # mark the event
    plural = f"{event_name.capitalize()}s"
    ax.set_xlabel(f"Lead time before {event_name} (steps)")
    ax.set_ylabel(f"{plural} caught ({pct})" if as_percent else f"Fraction of {event_name}s caught")
    if as_percent:
        ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize="small")
    return ax


def save_plot(path, *, fig=None, dpi=300, bbox_inches="tight") -> Path:
    """Save a figure to `path` (PDF if no extension given), with house settings."""
    fig = fig or plt.gcf()
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".pdf")
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Plot saved to: {path}")
    return path


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
