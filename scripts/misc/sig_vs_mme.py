#!/usr/bin/env python3
"""Signature kernel vs kernel mean embedding: empirical scaling law.

Simulates pairs of 1D OU processes with varying parameters, computes
the unnormalized signature kernel and empirical MME as a function of
trajectory length, then fits the power-law exponent alpha in
log(K_sig) ~ c * n^alpha and identifies how c depends on MME.

Usage:
    python sig_vs_mme.py
    python sig_vs_mme.py --t-max 20 --n-seeds 5
    python sig_vs_mme.py --plot-only
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sktime.dists_kernels import SignatureKernel

from utils.paths import get_output_dir
from utils.plotting import COL_WIDTH, FULL_WIDTH, setup_style

# Tol bright palette
COLORS = ["#4477AA", "#EE7733", "#228833", "#CC3311", "#66CCEE", "#AA3377", "#BBBBBB"]


def simulate_ou(theta, mu, sigma, dt, n_steps, rng):
    """Euler-Maruyama simulation of 1D OU process, initialized from stationary dist."""
    v = sigma ** 2 / (2 * theta)
    x = np.empty(n_steps)
    x[0] = rng.normal(mu, np.sqrt(v))
    sqrt_dt = np.sqrt(dt)
    noise = rng.standard_normal(n_steps - 1)
    for k in range(n_steps - 1):
        x[k + 1] = x[k] + theta * (mu - x[k]) * dt + sigma * sqrt_dt * noise[k]
    return x


def exact_mme(mu1, mu2, v1, v2, gamma):
    """Analytical kernel mean embedding inner product for two Gaussians with RBF kernel."""
    denom = 1 + 2 * gamma * (v1 + v2)
    return np.exp(-gamma * (mu1 - mu2) ** 2 / denom) / np.sqrt(denom)


def compute_empirical_mme(x, y, gamma, step_indices):
    """Empirical MME = (1/n^2) sum_{i,j} k_RBF(x_i, y_j) for each prefix length."""
    result = np.empty(len(step_indices))
    for i, n in enumerate(step_indices):
        diff = x[:n, None] - y[None, :n]
        result[i] = np.mean(np.exp(-gamma * diff ** 2))
    return result


def _make_unnorm_sig_kernel(gamma):
    """Create an unnormalized SignatureKernel with RBF static kernel."""
    def _fast_rbf(X, Y=None):
        if Y is None:
            Y = X
        X_sqnorms = np.sum(X ** 2, axis=1, keepdims=True)
        Y_sqnorms = np.sum(Y ** 2, axis=1, keepdims=True)
        dist_sq = X_sqnorms - 2 * X @ Y.T + Y_sqnorms.T
        np.maximum(dist_sq, 0, out=dist_sq)
        return np.exp(-gamma * dist_sq)

    return SignatureKernel(kernel=_fast_rbf, normalize=False)


def compute_log_sig_kernel(x, y, gamma, step_indices):
    """Unnormalized signature kernel for each prefix length.

    Returns log(K_sig(X_{0:n}, Y_{0:n})) for each n in step_indices.
    """
    k = _make_unnorm_sig_kernel(gamma)

    log_sig = np.empty(len(step_indices))
    for i, n in enumerate(step_indices):
        x_prefix = x[:n].reshape(1, n, 1).swapaxes(1, 2)  # (1, 1, n)
        y_prefix = y[:n].reshape(1, n, 1).swapaxes(1, 2)
        xy = np.concatenate([x_prefix, y_prefix], axis=0)
        K = k(xy)
        val = K[0, 1]
        log_sig[i] = np.log(val) if val > 0 else np.nan
    return log_sig


def run_experiment(params, gamma, dt, t_max, n_eval_points, n_seeds, base_seed):
    """Run sig kernel + MME computation for one pair of OU processes.

    params: (theta1, mu1, sigma1, theta2, mu2, sigma2)
    Returns: (step_indices, t_eval, log_sig_all, mme_all, mme_exact_val)
    """
    theta1, mu1, sigma1, theta2, mu2, sigma2 = params
    v1 = sigma1 ** 2 / (2 * theta1)
    v2 = sigma2 ** 2 / (2 * theta2)
    mme_exact_val = exact_mme(mu1, mu2, v1, v2, gamma)

    n_steps = int(t_max / dt)
    t_eval = np.linspace(t_max * 0.02, t_max, num=n_eval_points)
    step_indices = np.unique(np.round(t_eval / dt).astype(int))
    step_indices = step_indices[step_indices >= 20]
    t_eval = step_indices * dt

    log_sig_all = np.empty((n_seeds, len(step_indices)))
    mme_all = np.empty((n_seeds, len(step_indices)))

    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        x = simulate_ou(theta1, mu1, sigma1, dt, n_steps, rng)
        y = simulate_ou(theta2, mu2, sigma2, dt, n_steps, rng)

        mme_all[s] = compute_empirical_mme(x, y, gamma, step_indices)
        log_sig_all[s] = compute_log_sig_kernel(x, y, gamma, step_indices)

    return step_indices, t_eval, log_sig_all, mme_all, mme_exact_val


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t-max", type=float, default=10.0, help="Max continuous time")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-eval-points", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0, help="RBF kernel bandwidth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-only", action="store_true", help="Re-plot from cached data.npz")
    args = parser.parse_args()

    setup_style()
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.npz"

    # Different OU parameter pairs giving different MME values.
    # All use theta=5 (mixing time 0.2) and sigma=1.
    # Vary mu2 to get different stationary distribution separations.
    theta, sigma = 5.0, 1.0
    mu1 = 0.0
    mu2_values = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    param_sets = [(theta, mu1, sigma, theta, mu2, sigma) for mu2 in mu2_values]

    if args.plot_only:
        data = np.load(data_path, allow_pickle=True)
        all_results = data["all_results"].item()
        args.dt = float(data["dt"])
        args.gamma = float(data["gamma"])
        print(f"Loaded cached results")
    else:
        all_results = {}
        for i, params in enumerate(param_sets):
            mu2 = params[4]
            v = sigma ** 2 / (2 * theta)
            mme_val = exact_mme(mu1, mu2, v, v, args.gamma)
            print(f"\n{'='*60}")
            print(f"Config {i+1}/{len(param_sets)}: mu2={mu2}, exact MME={mme_val:.4f}")
            print(f"{'='*60}")

            step_indices, t_eval, log_sig_all, mme_all, mme_exact_val = run_experiment(
                params, args.gamma, args.dt, args.t_max,
                args.n_eval_points, args.n_seeds, args.seed,
            )
            all_results[mu2] = dict(
                step_indices=step_indices, t_eval=t_eval,
                log_sig_all=log_sig_all, mme_all=mme_all,
                mme_exact=mme_exact_val,
            )

            # Print progress
            mean_log_sig = log_sig_all.mean(axis=0)[-1]
            print(f"  log(K_sig)={mean_log_sig:.2f}, MME={mme_all.mean(axis=0)[-1]:.4f}")

        np.savez(data_path, all_results=all_results, dt=args.dt, gamma=args.gamma)
        print(f"\nCached results to {data_path}")

    # === Step 1: Fit alpha from log(log(K_sig)) vs log(n) ===
    print(f"\n{'='*60}")
    print("Step 1: Fit alpha in log(K_sig) ~ c * n^alpha")
    print(f"{'='*60}")

    fitted_alphas = []
    fitted_coeffs = []
    mme_values = []

    for i, (mu2, res) in enumerate(sorted(all_results.items())):
        n = res["step_indices"]
        log_sig_mean = res["log_sig_all"].mean(axis=0)

        # Use second half of data for fitting (past transient)
        half = len(n) // 2
        log_n = np.log(n[half:])
        log_log_sig = np.log(log_sig_mean[half:])

        slope, intercept, r, _, _ = linregress(log_n, log_log_sig)
        fitted_alphas.append(slope)
        fitted_coeffs.append(np.exp(intercept))
        mme_values.append(res["mme_exact"])

        print(f"  mu2={mu2}: alpha={slope:.3f}, c={np.exp(intercept):.4f}, "
              f"R²={r**2:.6f}, MME={res['mme_exact']:.4f}")

    mean_alpha = np.mean(fitted_alphas)
    print(f"\n  Mean alpha = {mean_alpha:.3f}")

    # === Step 2: How does c depend on MME? ===
    print(f"\n{'='*60}")
    print("Step 2: Fit c vs MME")
    print(f"{'='*60}")

    mme_arr = np.array(mme_values)
    c_arr = np.array(fitted_coeffs)

    # Try c ~ MME^beta
    log_mme = np.log(mme_arr)
    log_c = np.log(c_arr)
    beta_slope, beta_intercept, beta_r, _, _ = linregress(log_mme, log_c)
    print(f"  c ~ MME^beta: beta={beta_slope:.3f}, prefactor={np.exp(beta_intercept):.4f}, "
          f"R²={beta_r**2:.6f}")

    # === Step 3: Direct test of K_sig/T ~ sqrt(MME) ===
    # K_sig is huge, so work with log(K_sig/T) = log_sig - log(T)
    print(f"\n{'='*60}")
    print("Step 3: Direct test of K_sig/T vs sqrt(MME)")
    print(f"{'='*60}")

    log_sig_minus_logT = []
    sqrt_mme = []
    for mu2, res in sorted(all_results.items()):
        t_final = res["t_eval"][-1]
        val = res["log_sig_all"].mean(axis=0)[-1] - np.log(t_final)
        mme_val = res["mme_exact"]
        log_sig_minus_logT.append(val)
        sqrt_mme.append(np.sqrt(mme_val))
        print(f"  mu2={mu2}: log(K_sig/T)={val:.4f}, sqrt(MME)={np.sqrt(mme_val):.4f}")

    log_sig_minus_logT = np.array(log_sig_minus_logT)
    sqrt_mme = np.array(sqrt_mme)

    # Fit linear: log(K_sig/T) = a * sqrt(MME) + b
    # If K_sig/T ~ c * sqrt(MME), then log(K_sig/T) = log(c) + log(sqrt(MME))
    # But let's also try: log(K_sig/T) = a * sqrt(MME) + b  (linear in sqrt(MME))
    slope_h, intercept_h, r_h, _, _ = linregress(sqrt_mme, log_sig_minus_logT)
    print(f"\n  Linear fit: log(K_sig/T) = {slope_h:.3f} * sqrt(MME) + {intercept_h:.3f}")
    print(f"  R² = {r_h**2:.6f}")

    # Also try log-log: log(K_sig/T) vs log(sqrt(MME))
    slope_ll, intercept_ll, r_ll, _, _ = linregress(np.log(sqrt_mme), log_sig_minus_logT)
    print(f"  Power law: K_sig/T ~ exp({intercept_ll:.2f}) * sqrt(MME)^{slope_ll:.3f}")
    print(f"  R² = {r_ll**2:.6f}")

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.7))

    # Plot 1: log(log(K_sig)) vs log(n) — checking power law
    ax = axes[0, 0]
    for i, (mu2, res) in enumerate(sorted(all_results.items())):
        n = res["step_indices"]
        log_sig_mean = res["log_sig_all"].mean(axis=0)
        ax.plot(np.log(n), np.log(log_sig_mean), color=COLORS[i],
                linewidth=1.5, label=f"$\\mu_2={mu2}$")
    n_ref = np.array([n.min(), n.max()])
    ax.plot(np.log(n_ref), mean_alpha * np.log(n_ref) + np.mean(np.log(c_arr)),
            color="black", linestyle="--", linewidth=0.8,
            label=f"slope $= {mean_alpha:.2f}$")
    ax.set_xlabel(r"$\log n$")
    ax.set_ylabel(r"$\log(\log K_{\mathrm{sig}})$")
    ax.legend(fontsize=7, frameon=False)

    # Plot 2: log(K_sig) / n^alpha vs t — should flatten
    ax = axes[0, 1]
    for i, (mu2, res) in enumerate(sorted(all_results.items())):
        n = res["step_indices"]
        t = res["t_eval"]
        log_sig_mean = res["log_sig_all"].mean(axis=0)
        scaled = log_sig_mean / n ** mean_alpha
        ax.plot(t, scaled, color=COLORS[i], linewidth=1.5,
                label=f"$\\mu_2={mu2}$")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\log K_{\mathrm{sig}} \,/\, n^{\hat{\alpha}}$")
    ax.legend(fontsize=7, frameon=False)

    # Plot 3: fitted c vs MME
    ax = axes[1, 0]
    ax.scatter(mme_arr, c_arr, color=COLORS[0], s=30, zorder=5)
    mme_fit = np.linspace(mme_arr.min() * 0.8, mme_arr.max() * 1.2, 50)
    ax.plot(mme_fit, np.exp(beta_intercept) * mme_fit ** beta_slope,
            color="black", linestyle="--", linewidth=0.8,
            label=f"$c \\propto \\mathrm{{MME}}^{{{beta_slope:.2f}}}$")
    ax.set_xlabel("Exact MME")
    ax.set_ylabel(r"Fitted $c$")
    ax.legend(fontsize=7, frameon=False)

    # Plot 4: log(K_sig/T) vs sqrt(MME) — direct hypothesis test
    ax = axes[1, 1]
    ax.scatter(sqrt_mme, log_sig_minus_logT, color=COLORS[0], s=30, zorder=5)
    x_fit = np.linspace(0, sqrt_mme.max() * 1.1, 50)
    ax.plot(x_fit, slope_h * x_fit + intercept_h,
            color="black", linestyle="--", linewidth=0.8,
            label=f"$y = {slope_h:.1f}x {intercept_h:+.1f}$, $R^2={r_h**2:.3f}$")
    ax.set_xlabel(r"$\sqrt{\mathrm{MME}}$")
    ax.set_ylabel(r"$\log(K_{\mathrm{sig}} \,/\, T)$")
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    out_path = output_dir / "scaling.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved to {out_path}")

    # === Convergence plot: 3 subplots for a representative config ===
    # Use the fitted power law: K_sig/T ~ C * MME^2, i.e. log(K_sig) ~ 2*log(MME) + log(T) + const
    # Pick mu2=1.0 as representative
    ref_key = 1.0
    if ref_key not in all_results:
        ref_key = sorted(all_results.keys())[len(all_results) // 2]
    res = all_results[ref_key]
    t = res["t_eval"]
    mme_exact_ref = res["mme_exact"]

    def _plot_seeds(ax, t, data, color):
        for s in range(data.shape[0]):
            ax.plot(t, data[s], color=color, alpha=0.2, linewidth=0.7)
        ax.plot(t, data.mean(axis=0), color=color, linewidth=1.5)

    fig, axes = plt.subplots(3, 1, figsize=(COL_WIDTH, COL_WIDTH * 1.6), sharex=True)

    # Top: empirical + exact MME
    ax = axes[0]
    _plot_seeds(ax, t, res["mme_all"], "#EE7733")
    ax.axhline(mme_exact_ref, color="black", linestyle="--", linewidth=1.0,
               label="Exact MME")
    ax.set_ylabel("MME")
    ax.legend(frameon=False)

    # Middle: log(K_sig) vs predicted log(T) + const
    ax = axes[1]
    _plot_seeds(ax, t, res["log_sig_all"], "#4477AA")
    # Predicted: log(K_sig) = log(T) + 2*log(MME) + const
    # Use intercept_ll from the power law fit
    predicted = np.log(t) + slope_ll * np.log(np.sqrt(mme_exact_ref)) + intercept_ll
    ax.plot(t, predicted, color="black", linestyle="--", linewidth=1.0,
            label=r"$\log T + 2\log\mathrm{MME} + C$")
    ax.set_ylabel(r"$\log K_{\mathrm{sig}}$")
    ax.legend(fontsize=8, frameon=False)

    # Bottom: K_sig / (T * MME^2) — should be constant
    # log ratio = log_sig - log(T) - 2*log(MME)
    log_ratio = res["log_sig_all"] - np.log(t)[None, :] - 2 * np.log(mme_exact_ref)
    ratio = np.exp(log_ratio)
    ax = axes[2]
    _plot_seeds(ax, t, ratio, "#66CCEE")
    ax.set_yscale("log")
    ax.set_ylabel(r"$K_{\mathrm{sig}} \,/\, (T \cdot \mathrm{MME}^2)$")
    ax.set_xlabel("Time $t$")

    fig.suptitle(f"$\\mu_2 = {ref_key}$, MME $= {mme_exact_ref:.3f}$", fontsize=10)
    fig.tight_layout()
    out_path = output_dir / "convergence.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
