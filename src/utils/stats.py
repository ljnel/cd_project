import numpy as np


def trajectory_stats(states: np.ndarray, dim_names: list[str] | None = None) -> dict:
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


def mmd_squared(K_in: np.ndarray, K_out: np.ndarray, K_cross: np.ndarray) -> np.ndarray:
    """Unbiased MMD^2 from kernel matrices. All inputs (T, *, *). Returns (T,)."""
    T = K_in.shape[0]
    n = K_in.shape[1]
    m = K_out.shape[1]

    K_in = K_in.copy()
    K_out = K_out.copy()
    for t in range(T):
        np.fill_diagonal(K_in[t], 0)
        np.fill_diagonal(K_out[t], 0)

    term_in = K_in.sum(axis=(1, 2)) / (n * (n - 1))
    term_out = K_out.sum(axis=(1, 2)) / (m * (m - 1))
    term_cross = K_cross.sum(axis=(1, 2)) / (n * m)

    return term_in + term_out - 2 * term_cross


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
    from algs.spatial_kernel import (
        fit_gammas, fit_rbf_gamma, humanoid_kernel_matrices, rbf_kernel_matrices,
    )

    y = np.asarray(y, dtype=bool)
    x_in = x[~y]
    x_out = x[y]
    N_in, N_out = x_in.shape[0], x_out.shape[0]
    T = x.shape[1]

    if not kernel_kwargs:
        kernel_kwargs = fit_gammas(x_in)
    if rbf_gamma is None:
        rbf_gamma = fit_rbf_gamma(x_in)

    results = {}
    for name, kern_fn in [
        ("composite", lambda a, b=None: humanoid_kernel_matrices(a, b, **kernel_kwargs)),
        ("rbf", lambda a, b=None: rbf_kernel_matrices(a, b, gamma=rbf_gamma)),
    ]:
        K_in = kern_fn(x_in)
        K_out = kern_fn(x_out)
        K_cross = kern_fn(x_in, x_out)
        results[name] = mmd_squared(K_in, K_out, K_cross)

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
