#!/usr/bin/env python3
import logging
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore")

from cd.envs.info import ENV_INFO
from cd.utils.paths import get_output_dir, get_root
from cd.utils.plotting import FULL_WIDTH, setup_style
from cd.utils.stats import mmd_squared

logger = logging.getLogger("cd.scripts.failure_mode_diagnostics")

EnvName = Literal['ant', 'half_cheetah', 'hopper', 'humanoid', 'inv_pend', 'upkie']
Hypothesis = Literal['A', 'B', 'C']
ALL_ENVS: list[EnvName] = ['ant', 'half_cheetah', 'hopper', 'humanoid', 'inv_pend', 'upkie']
NO_SKIP: list[Hypothesis] = []
UNSTABLE = {'ant', 'half_cheetah'}
DELTAS_STEPS = tuple(range(5, 101, 5))

ENV_COLORS = {
    'ant':          '#CC3311',
    'half_cheetah': '#EE7733',
    'hopper':       '#0077BB',
    'humanoid':     '#009988',
    'inv_pend':     '#33BBEE',
    'upkie':        '#AA4499',
}


def _ordered(envs):
    """Stable group first (alpha), then unstable group (alpha)."""
    stable = sorted(e for e in envs if e not in UNSTABLE)
    unstable = sorted(e for e in envs if e in UNSTABLE)
    return stable + unstable


def _label(env):
    return ENV_INFO[env].display_name


def _slice_obs(X, env):
    s = ENV_INFO[env].obs_slice
    return X[..., s] if s is not None else X


def _load_npz(env, name):
    p = get_root() / 'data' / env / name / 'data.npz'
    return np.load(p, allow_pickle=False)


def _params(d):
    return np.stack(
        [d['mass_scale'], d['friction_scale'], d['damping_scale']],
        axis=1,
    ).astype(np.float32)


# ============================================================================
# Hypothesis A: Horizon (fail-step distribution)
# ============================================================================

def compute_horizon(env):
    d = _load_npz(env, 'fail_pred')
    fail = d['fail'].astype(np.int64)
    fs = fail[fail >= 0]
    return {
        'fail_steps': fs,
        'n_total': len(fail),
        'n_fail': len(fs),
    }


def plot_horizon(per_env, out_path, log_x=True):
    envs = _ordered(per_env.keys())
    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 2.8),
        gridspec_kw={'width_ratios': [1.1, 1]},
    )

    # Panel 1: violin of fail-time (s)
    ax = axes[0]
    data, labels, colors = [], [], []
    for env in envs:
        r = per_env[env]
        if r['n_fail'] == 0:
            continue
        data.append(r['fail_steps'])
        labels.append(_label(env))
        colors.append(ENV_COLORS[env])

    parts = ax.violinplot(data, vert=False, showmedians=True, widths=0.85)
    for body, c in zip(parts['bodies'], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.6)
    for k in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if k in parts:
            parts[k].set_color('0.3')
            parts[k].set_linewidth(0.8)
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Failure step')
    if log_x:
        ax.set_xscale('log')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_title('(a) Fail-time distribution', fontsize=10, loc='left')

    # Panel 2: cumulative failure rate vs time
    ax = axes[1]
    for env in envs:
        r = per_env[env]
        if r['n_fail'] == 0:
            continue
        sorted_fs = np.sort(r['fail_steps'])
        y = np.arange(1, len(sorted_fs) + 1) / r['n_total']
        ts = np.concatenate([[0.5], sorted_fs])
        ys = np.concatenate([[0], y])
        ls = '-' if env in UNSTABLE else '--'
        ax.plot(
            ts, ys, drawstyle='steps-post',
            color=ENV_COLORS[env], label=_label(env),
            linewidth=1.2, linestyle=ls,
        )
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$P(\mathrm{fail\ before\ } t)$')
    if log_x:
        ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title('(b) Cumulative failure rate', fontsize=10, loc='left')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ============================================================================
# Hypothesis B: Separability (kernel MMD)
# ============================================================================

def _mmd_z_with_ci(K, n_f, n_s, n_perms, n_boots, rng):
    """MMD z-score against permutation null + 90% bootstrap CI via 80% subsample.

    K is the (n_f + n_s) x (n_f + n_s) Gram matrix of pooled samples,
    with the first n_f rows/cols indexing failures, the rest successes.
    """
    y = np.concatenate([np.ones(n_f, dtype=int), np.zeros(n_s, dtype=int)])
    mmd2 = mmd_squared(K, y)

    null = np.empty(n_perms)
    for p in range(n_perms):
        null[p] = mmd_squared(K, rng.permutation(y))
    null_mean, null_std = null.mean(), null.std()
    if null_std == 0:
        null_std = 1.0
    z = (mmd2 - null_mean) / null_std

    # 80% subsample-without-replacement bootstrap to avoid kernel-diagonal repeats
    n_pairs = min(n_f, n_s)
    n_sub = max(30, int(0.8 * n_pairs))
    z_boots = np.empty(n_boots)
    for b in range(n_boots):
        idx_f = rng.choice(n_f, n_sub, replace=False)
        idx_s = rng.choice(n_s, n_sub, replace=False)
        idx = np.concatenate([idx_f, n_f + idx_s])
        K_b = K[np.ix_(idx, idx)]
        y_b = np.concatenate([np.ones(n_sub, dtype=int), np.zeros(n_sub, dtype=int)])
        z_boots[b] = (mmd_squared(K_b, y_b) - null_mean) / null_std
    z_lo, z_hi = np.percentile(z_boots, [5, 95])
    return z, z_lo, z_hi


def compute_separability(env, deltas_steps=DELTAS_STEPS, max_pairs=500,
                         n_perms=200, n_boots=200, seed=42):
    rng = np.random.default_rng(seed)
    d = _load_npz(env, 'fail_pred')
    X = _slice_obs(d['X'], env).astype(np.float32)
    fail = d['fail'].astype(np.int64)
    params = _params(d)

    fail_eps = np.where(fail >= 0)[0]
    succ_eps = np.where(fail < 0)[0]
    if len(fail_eps) < 30 or len(succ_eps) < 30:
        logger.warning(f"  {env}: too few fail/succ episodes ({len(fail_eps)}/{len(succ_eps)})")
        return None

    # Standardize obs per dim using pooled stats
    flat = X.reshape(-1, X.shape[-1])
    mu, sigma = flat.mean(0), flat.std(0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    Xs = (X - mu) / sigma

    # Per-failure NN match in standardized param space among successes
    p_mu, p_sigma = params.mean(0), params.std(0)
    p_sigma = np.where(p_sigma > 0, p_sigma, 1.0)
    p_z = (params - p_mu) / p_sigma
    tree = BallTree(p_z[succ_eps])
    _, idx = tree.query(p_z[fail_eps], k=1)
    matched_succ_eps = succ_eps[idx[:, 0]]

    fail_t = fail[fail_eps].astype(int)
    results = {}
    for delta_steps in deltas_steps:
        valid = fail_t >= delta_steps
        eps_f = fail_eps[valid]
        eps_s = matched_succ_eps[valid]
        t_f = fail_t[valid]

        n = len(eps_f)
        if n > max_pairs:
            sub = rng.choice(n, max_pairs, replace=False)
            eps_f, eps_s, t_f = eps_f[sub], eps_s[sub], t_f[sub]
            n = max_pairs
        if n < 30:
            results[int(delta_steps)] = None
            continue

        target_t = t_f - delta_steps
        x_f = Xs[eps_f, target_t]
        x_s = Xs[eps_s, target_t]

        # Median-bandwidth Gaussian kernel on pooled samples
        pooled = np.vstack([x_f, x_s])
        sub = pooled if len(pooled) <= 1000 else pooled[rng.choice(len(pooled), 1000, replace=False)]
        pdist = pairwise_distances(sub)
        med = np.median(pdist[pdist > 0])
        sigma2 = max(med ** 2, 1e-12)
        K = np.exp(-pairwise_distances(pooled, squared=True) / (2 * sigma2))

        z, z_lo, z_hi = _mmd_z_with_ci(K, n, n, n_perms, n_boots, rng)
        results[int(delta_steps)] = {
            'z': z, 'z_lo': z_lo, 'z_hi': z_hi, 'n_pairs': n,
            'sigma2': sigma2,
        }

    return {
        'per_delta': results,
        'deltas_steps': np.array(deltas_steps, dtype=int),
    }


def plot_separability(per_env, out_path, log_y=False):
    envs = _ordered(per_env.keys())
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, 3.0))
    ax.axhline(0, color='0.5', linewidth=0.5, linestyle=':')
    for env in envs:
        r = per_env[env]
        if r is None:
            continue
        deltas = r['deltas_steps']
        z = np.array([
            r['per_delta'][int(d)]['z'] if r['per_delta'].get(int(d)) else np.nan
            for d in deltas
        ])
        z_lo = np.array([
            r['per_delta'][int(d)]['z_lo'] if r['per_delta'].get(int(d)) else np.nan
            for d in deltas
        ])
        z_hi = np.array([
            r['per_delta'][int(d)]['z_hi'] if r['per_delta'].get(int(d)) else np.nan
            for d in deltas
        ])
        m = ~np.isnan(z)
        ls = '-' if env in UNSTABLE else '--'
        ax.plot(
            deltas[m], z[m],
            color=ENV_COLORS[env], label=_label(env),
            linewidth=1.4, linestyle=ls, marker='o', markersize=3,
        )
        ax.fill_between(
            deltas[m], z_lo[m], z_hi[m],
            color=ENV_COLORS[env], alpha=0.15, linewidth=0,
        )
    ax.set_xlabel(r'Lookback $\Delta$ before failure (steps)')
    ax.set_ylabel(r'MMD $z$-score (vs.\ permutation null)')
    ax.set_xticks([5, 25, 50, 75, 100])
    if log_y:
        ax.set_yscale('symlog', linthresh=1)
    ax.legend(loc='upper right', frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ============================================================================
# Hypothesis C: Local divergence rate (Rosenstein-style on success-only)
# ============================================================================

def compute_divergence(env, t_burnin=100, tau=50, n_anchors=1000,
                       t_window=5, param_tol_frac=0.10, n_boots=200, seed=42):
    rng = np.random.default_rng(seed)
    d = _load_npz(env, 'train')
    X = _slice_obs(d['X'], env).astype(np.float32)
    params = _params(d)

    N, T, dim = X.shape
    if t_burnin + tau >= T:
        logger.warning(f"  {env}: t_burnin ({t_burnin}) + tau ({tau}) >= T ({T})")
        return None

    # Standardize obs per dim using pooled stats
    flat = X.reshape(-1, dim)
    mu, sigma = flat.mean(0), flat.std(0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    Xs = ((X - mu) / sigma).astype(np.float32)

    # Param tolerance: ±10% of per-dim range, with 0-range fallback to 1.0
    p_range = params.max(0) - params.min(0)
    p_tol = param_tol_frac * np.where(p_range > 0, p_range, 1.0)

    # Episode-level param-match matrix (N, N)
    p_diff = np.abs(params[:, None, :] - params[None, :, :])
    ep_match = np.all(p_diff <= p_tol, axis=2)
    np.fill_diagonal(ep_match, False)  # exclude same-episode neighbors

    # Sample anchors uniformly from valid (ep, t) range
    n_valid_t = T - tau - t_burnin
    anchor_eps = rng.choice(N, size=n_anchors, replace=True)
    anchor_ts = rng.choice(n_valid_t, size=n_anchors, replace=True) + t_burnin

    log_d = np.full((n_anchors, tau + 1), np.nan, dtype=np.float64)
    for i in range(n_anchors):
        ep_a = int(anchor_eps[i])
        t_a = int(anchor_ts[i])
        valid_eps = np.where(ep_match[ep_a])[0]
        if len(valid_eps) == 0:
            continue
        t_lo = max(t_burnin, t_a - t_window)
        t_hi = min(T - tau, t_a + t_window + 1)
        if t_hi <= t_lo:
            continue
        valid_ts = np.arange(t_lo, t_hi)
        # NN within (valid_eps x valid_ts) grid
        cand = Xs[valid_eps[:, None], valid_ts[None, :]]  # (K, M, dim)
        anchor = Xs[ep_a, t_a]                            # (dim,)
        dists = np.linalg.norm(cand - anchor, axis=2)     # (K, M)
        j = int(np.argmin(dists))
        k_idx, m_idx = np.unravel_index(j, dists.shape)
        ep_b = int(valid_eps[k_idx])
        t_b = int(valid_ts[m_idx])
        # Forward distance trajectory
        traj = np.linalg.norm(
            Xs[ep_b, t_b:t_b + tau + 1] - Xs[ep_a, t_a:t_a + tau + 1],
            axis=1,
        )
        log_d[i] = np.log(traj + 1e-12)

    valid = ~np.isnan(log_d[:, 0])
    n_valid = int(valid.sum())
    if n_valid < 30:
        logger.warning(f"  {env}: only {n_valid} valid anchors")
        return None
    log_d_v = log_d[valid]

    s_steps = np.arange(tau + 1)
    s_fit = s_steps[2:]              # skip s=0,1 (NN-discretization noise)
    y = log_d_v.mean(0)
    slope, _ = np.polyfit(s_fit, y[2:], 1)

    boot = np.empty(n_boots)
    for b in range(n_boots):
        idx = rng.choice(n_valid, size=n_valid, replace=True)
        y_b = log_d_v[idx].mean(0)
        boot[b], _ = np.polyfit(s_fit, y_b[2:], 1)
    slope_lo, slope_hi = np.percentile(boot, [5, 95])

    return {
        'y': y,
        's_steps': s_steps,
        'slope_per_step': slope,
        'slope_per_step_lo': slope_lo,
        'slope_per_step_hi': slope_hi,
        'n_valid_anchors': n_valid,
        'tau': tau,
    }


def plot_divergence(per_env, out_path):
    envs = _ordered(per_env.keys())
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, 3.0))
    for env in envs:
        r = per_env[env]
        if r is None:
            continue
        slope = r['slope_per_step']
        lo, hi = r['slope_per_step_lo'], r['slope_per_step_hi']
        label = (f"{_label(env)}: "
                 f"$\\hat\\lambda = {slope:+.3f}\\,$"
                 f"[{lo:+.3f}, {hi:+.3f}] nats/step")
        ls = '-' if env in UNSTABLE else '--'
        ax.plot(r['s_steps'], r['y'], color=ENV_COLORS[env], label=label,
                linewidth=1.3, linestyle=ls)
    ax.set_xlabel(r'Lag $s$ (steps)')
    ax.set_ylabel(r'$\overline{\log d(s)}$')
    ax.legend(loc='lower right', frameon=False, fontsize=6.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ============================================================================
# Caching
# ============================================================================

def _flatten_for_npz(payload):
    """Flatten nested dict-of-dicts into a single dict of arrays for np.savez."""
    out = {}
    # Hypothesis A
    for env, r in payload.get('A', {}).items():
        out[f'A__{env}__fail_steps'] = r['fail_steps']
        out[f'A__{env}__n_total'] = np.array(r['n_total'])
        out[f'A__{env}__n_fail'] = np.array(r['n_fail'])
    # Hypothesis B
    for env, r in payload.get('B', {}).items():
        if r is None:
            continue
        out[f'B__{env}__deltas_steps'] = r['deltas_steps']
        zs = []
        zlo = []
        zhi = []
        npairs = []
        for delta in r['deltas_steps']:
            cell = r['per_delta'].get(int(delta))
            if cell is None:
                zs.append(np.nan); zlo.append(np.nan); zhi.append(np.nan); npairs.append(0)
            else:
                zs.append(cell['z']); zlo.append(cell['z_lo']); zhi.append(cell['z_hi'])
                npairs.append(cell['n_pairs'])
        out[f'B__{env}__z'] = np.array(zs)
        out[f'B__{env}__z_lo'] = np.array(zlo)
        out[f'B__{env}__z_hi'] = np.array(zhi)
        out[f'B__{env}__n_pairs'] = np.array(npairs)
    # Hypothesis C
    for env, r in payload.get('C', {}).items():
        if r is None:
            continue
        out[f'C__{env}__y'] = r['y']
        out[f'C__{env}__s_steps'] = r['s_steps']
        out[f'C__{env}__slope_per_step'] = np.array(r['slope_per_step'])
        out[f'C__{env}__slope_per_step_lo'] = np.array(r['slope_per_step_lo'])
        out[f'C__{env}__slope_per_step_hi'] = np.array(r['slope_per_step_hi'])
        out[f'C__{env}__n_valid_anchors'] = np.array(r['n_valid_anchors'])
        out[f'C__{env}__tau'] = np.array(r['tau'])
    return out


def _unflatten_from_npz(d):
    payload = {'A': {}, 'B': {}, 'C': {}}
    keys = list(d.files)
    for env in ALL_ENVS:
        if f'A__{env}__fail_steps' in keys:
            payload['A'][env] = {
                'fail_steps': d[f'A__{env}__fail_steps'],
                'n_total': int(d[f'A__{env}__n_total']),
                'n_fail': int(d[f'A__{env}__n_fail']),
            }
        if f'B__{env}__z' in keys:
            zs = d[f'B__{env}__z']
            zlo = d[f'B__{env}__z_lo']
            zhi = d[f'B__{env}__z_hi']
            npairs = d[f'B__{env}__n_pairs']
            ds = d[f'B__{env}__deltas_steps']
            per_delta = {}
            for i, delta in enumerate(ds):
                if np.isnan(zs[i]):
                    per_delta[int(delta)] = None
                else:
                    per_delta[int(delta)] = {
                        'z': float(zs[i]), 'z_lo': float(zlo[i]),
                        'z_hi': float(zhi[i]), 'n_pairs': int(npairs[i]),
                    }
            payload['B'][env] = {
                'per_delta': per_delta,
                'deltas_steps': ds,
            }
        if f'C__{env}__y' in keys:
            payload['C'][env] = {
                'y': d[f'C__{env}__y'],
                's_steps': d[f'C__{env}__s_steps'],
                'slope_per_step': float(d[f'C__{env}__slope_per_step']),
                'slope_per_step_lo': float(d[f'C__{env}__slope_per_step_lo']),
                'slope_per_step_hi': float(d[f'C__{env}__slope_per_step_hi']),
                'n_valid_anchors': int(d[f'C__{env}__n_valid_anchors']),
                'tau': int(d[f'C__{env}__tau']),
            }
    return payload


# ============================================================================
# Main
# ============================================================================

def main(
    envs: list[EnvName] = ALL_ENVS,
    skip: list[Hypothesis] = NO_SKIP,
    no_cache: bool = False,
    seed: int = 42,
):
    """Failure-mode diagnostics across environments.

    Investigates why CD/kernel-CD detection achieves near-perfect AUC on
    hopper/humanoid/upkie/inv_pend but fails on ant/half_cheetah, testing three
    hypotheses (B is the lead): A. Horizon (failures happen too early to
    accumulate evidence); B. Separability (pre-failure obs are statistically
    indistinguishable from comparable success obs); C. Local divergence (high
    local divergence rate on the success attractor widens the training
    distribution). Writes horizon{,_linear}.pdf, separability{,_log}.pdf,
    divergence.pdf, and a data.npz cache to outputs/failure_mode_diagnostics/.

    Args:
        envs: Environments to include.
        skip: Hypotheses to skip (any of A, B, C).
        no_cache: Recompute even if data.npz exists.
        seed: Random seed.
    """
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    setup_style()

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / 'data.npz'

    payload = {'A': {}, 'B': {}, 'C': {}}
    if cache_path.exists() and not no_cache:
        logger.info(f"Loading cache from {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        payload = _unflatten_from_npz(cached)

    # Hypothesis A
    if 'A' not in skip:
        for env in envs:
            if env in payload['A']:
                logger.info(f"[A] {env}: from cache")
                continue
            logger.info(f"[A] {env}: computing")
            payload['A'][env] = compute_horizon(env)
        plot_horizon({e: payload['A'][e] for e in envs if e in payload['A']},
                     out_dir / 'horizon.pdf')
        plot_horizon({e: payload['A'][e] for e in envs if e in payload['A']},
                     out_dir / 'horizon_linear.pdf', log_x=False)

    # Hypothesis B
    if 'B' not in skip:
        for env in envs:
            if env in payload['B']:
                logger.info(f"[B] {env}: from cache")
                continue
            logger.info(f"[B] {env}: computing MMD across {len(DELTAS_STEPS)} deltas")
            payload['B'][env] = compute_separability(env, deltas_steps=DELTAS_STEPS, seed=seed)
        plot_separability({e: payload['B'].get(e) for e in envs},
                          out_dir / 'separability.pdf')
        plot_separability({e: payload['B'].get(e) for e in envs},
                          out_dir / 'separability_log.pdf', log_y=True)

    # Hypothesis C
    if 'C' not in skip:
        for env in envs:
            if env in payload['C']:
                logger.info(f"[C] {env}: from cache")
                continue
            logger.info(f"[C] {env}: computing local divergence rate")
            payload['C'][env] = compute_divergence(env, seed=seed)
        plot_divergence({e: payload['C'].get(e) for e in envs},
                        out_dir / 'divergence.pdf')

    # Save cache
    flat = _flatten_for_npz(payload)
    np.savez(cache_path, **flat)
    logger.info(f"Cached numerics to {cache_path}")


if __name__ == '__main__':
    tyro.cli(main)
