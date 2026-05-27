import numpy as np


def channel_stats(states: np.ndarray, dim_names: list[str] | None = None) -> dict:
    """Per-dimension stats for trajectory data of shape (N, T, D)."""
    N, T, D = states.shape
    flat = states.reshape(-1, D)  # (N * T, D)

    names = dim_names or [f"dim_{i}" for i in range(D)]
    stats = {}
    for i, name in enumerate(names):
        col = flat[:, i]
        stats[name] = {
            "mean": col.mean(),
            "std": col.std(),
            "min": col.min(),
            "max": col.max(),
            "q25": np.percentile(col, 25),
            "median": np.median(col),
            "q75": np.percentile(col, 75),
        }

    header = f"{'dim':<16} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'median':>9}"
    print(f"Trajectories: {N} episodes x {T} steps x {D} dims")
    print(header)
    print("-" * len(header))
    for name, s in stats.items():
        print(f"{name:<16} {s['mean']:>9.3f} {s['std']:>9.3f} {s['min']:>9.3f} {s['max']:>9.3f} {s['median']:>9.3f}")

    return stats


def mmd_squared(K: np.ndarray, y: np.ndarray) -> np.ndarray | float:
    """Unbiased MMD^2 from a Gram matrix and binary labels.

    Args:
        K: Gram matrix, shape (N, N) or (T, N, N).
        y: binary labels (N,), 0 = inlier, 1 = outlier.

    Returns:
        Scalar if K is 2D, array of shape (T,) if K is 3D.
    """
    assert np.isin(y, [0, 1]).all(), f"y must be binary, got unique values {np.unique(y)}"
    batched = K.ndim == 3
    if not batched:
        K = K[None]  # (1, N, N)

    mask_in = y == 0
    mask_out = y == 1
    n = mask_in.sum()
    m = mask_out.sum()

    ix_in = np.where(mask_in)[0]
    ix_out = np.where(mask_out)[0]

    K_in = K[:, ix_in[:, None], ix_in[None, :]].copy() if n > 1 else np.zeros((K.shape[0], 0, 0))
    K_out = K[:, ix_out[:, None], ix_out[None, :]].copy() if m > 1 else np.zeros((K.shape[0], 0, 0))
    K_cross = K[:, ix_in[:, None], ix_out[None, :]]

    # Zero diagonals for unbiased estimate
    for t in range(K.shape[0]):
        if n > 1:
            np.fill_diagonal(K_in[t], 0)
        if m > 1:
            np.fill_diagonal(K_out[t], 0)

    term_in = K_in.sum(axis=(1, 2)) / (n * (n - 1)) if n > 1 else 0.0
    term_out = K_out.sum(axis=(1, 2)) / (m * (m - 1)) if m > 1 else 0.0
    term_cross = K_cross.sum(axis=(1, 2)) / (n * m) if n > 0 and m > 0 else 0.0

    result = term_in + term_out - 2 * term_cross
    return result if batched else result.item()


def mmd_kernel_comparison(
    x: np.ndarray,
    y: np.ndarray,
    rbf_gamma: float | None = None,
    **kernel_kwargs,
) -> dict:
    """Per-timestep MMD^2 between inliers/outliers for composite vs RBF kernel.

    Args:
        x: (N, T, D) trajectory data
        y: (N,) binary, 0 = inlier, 1 = outlier
        rbf_gamma: bandwidth for RBF kernel (default: 1/D)
        **kernel_kwargs: forwarded to humanoid_composite_kernel
    """
    from algs.kernels.spatial_kernel import (
        fit_gammas, fit_rbf_gamma, humanoid_kernel_matrices, rbf_kernel_matrices,
    )

    y = np.asarray(y, dtype=int)
    N_in, N_out = (y == 0).sum(), (y == 1).sum()
    T = x.shape[1]

    x_in = x[y == 0]
    if not kernel_kwargs:
        kernel_kwargs = fit_gammas(x_in)
    if rbf_gamma is None:
        rbf_gamma = fit_rbf_gamma(x_in)

    results = {}
    for name, kern_fn in [
        ("composite", lambda a, b=None: humanoid_kernel_matrices(a, b, **kernel_kwargs)),
        ("rbf", lambda a, b=None: rbf_kernel_matrices(a, b, gamma=rbf_gamma)),
    ]:
        K = kern_fn(x)  # (T, N, N)
        results[name] = mmd_squared(K, y)

    header = f"{'step':>6} {'composite':>12} {'rbf':>12} {'ratio':>8}"
    print(f"MMD^2 per timestep (N_in={N_in}, N_out={N_out})")
    print(header)
    print("-" * len(header))
    step_indices = np.linspace(0, T - 1, min(T, 20), dtype=int)
    for t in step_indices:
        c, r = results["composite"][t], results["rbf"][t]
        ratio = c / r if r > 0 else float("inf")
        print(f"{t:>6} {c:>12.6f} {r:>12.6f} {ratio:>8.2f}")

    print(f"\n{'mean':>6} {results['composite'].mean():>12.6f} {results['rbf'].mean():>12.6f}")

    return results


def twonn(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """Estimate intrinsic dimension via Two-NN (Facco et al., 2017).

    Args:
        X: (N, D) point cloud. For trajectory data of shape (N, T, D),
            flatten first: `X.reshape(-1, D)`.
        discard_fraction: top-tail fraction of mu = r2/r1 to discard before
            the linear fit (robust to near-duplicate points).
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1, r2 = dists[:, 1], dists[:, 2]

    mu = np.sort(r2[r1 > 0] / r1[r1 > 0])
    n = int(len(mu) * (1 - discard_fraction))
    mu = mu[:n]

    F = np.arange(1, n + 1) / (n + 1)
    x = np.log(mu)
    y = -np.log(1 - F)
    return float(np.dot(x, y) / np.dot(x, x))
