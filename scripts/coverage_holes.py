#!/usr/bin/env python3
import warnings
from typing import Literal

import numpy as np
import tyro

warnings.filterwarnings("ignore")

import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import SAC

from data.configs import DATASETS
from data.dataset import survived
from data.io import load
from detectors.cd_poly import CDPolyDetector
from envs.info import ENV_INFO
from eval.calibration import max_conformal_threshold
from utils.paths import get_output_dir, get_root

EnvName = Literal['ant', 'half_cheetah', 'hopper', 'humanoid', 'inv_pend', 'upkie']

DEFAULT_DEGREES = [2]
DEFAULT_SIGMAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

VEL_CLIP = 9.99   # |qvel| at/above this is clipped in the obs -> not reconstructable
EARLY_K = 50      # a failure before this many steps counts as "early"


def pooled_states(episodes: np.ndarray) -> np.ndarray:
    """Flatten (n_eps, T, D) to (n_states, D), dropping any non-finite rows."""
    flat = episodes.reshape(-1, episodes.shape[-1])
    return flat[np.isfinite(flat).all(axis=1)]


def score_in_batches(detector, x: np.ndarray, batch: int = 20_000) -> np.ndarray:
    """CD score for a large point set, batched to bound peak memory."""
    return np.concatenate(
        [detector.score(x[i:i + batch]) for i in range(0, len(x), batch)]
    )


def calibrate_threshold(detector, cal_norm: np.ndarray, n_eps: int, T: int,
                        alpha: float) -> float:
    """Per-trajectory max-conformal threshold.

    `cal_norm` is the channel-normalized calibration states flattened to
    (n_eps*T, D). Score them, reshape to (n_eps, T), and hand the matrix to
    `max_conformal_threshold`, which takes the per-episode max and the
    conformal quantile — the smallest tau for which ~(1 - alpha) of safe
    trajectories stay below it at every step.
    """
    flat_scores = score_in_batches(detector, cal_norm)
    return max_conformal_threshold(flat_scores.reshape(n_eps, T), alpha)


def healthy_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim (low, high) healthy box in observation coordinates.

    Hopper obs = [z, torso angle, 3 joint angles, 6 vels]; is_healthy requires
    z in z_range, torso angle in angle_range, and every dim of obs[1:] in
    state_range. z (obs[0]) is governed only by z_range.
    """
    u = env.unwrapped
    D = u.observation_space.shape[0]
    lo = np.full(D, u._healthy_state_range[0], dtype=float)
    hi = np.full(D, u._healthy_state_range[1], dtype=float)
    lo[0], hi[0] = u._healthy_z_range
    lo[1] = max(lo[1], u._healthy_angle_range[0])
    hi[1] = min(hi[1], u._healthy_angle_range[1])
    return lo, hi


def is_healthy(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Row-wise healthy mask for raw observations."""
    return ((obs >= lo) & (obs <= hi)).all(axis=-1)


def reconstructable(obs: np.ndarray, vel_start: int) -> np.ndarray:
    """Row-wise mask: velocities strictly inside the clip, so obs -> sim state
    is faithful (Hopper clips qvel to +/-10 in the observation)."""
    return (np.abs(obs[..., vel_start:]) < VEL_CLIP).all(axis=-1)


def perturb_sample(detector, scaler, base_norm, sigma, tau, lo, hi, vel_start,
                   n_target, rng, batch=8192, max_proposals=5_000_000):
    """On-manifold proposals: perturb random base states (in normalized space)
    by N(0, sigma^2 I), keep those still in the set (cd <= tau), healthy, and
    faithfully reconstructable.

    Returns the accepted *raw* starting points (capped at n_target), the number
    proposed, and the number accepted (before the cap).
    """
    accepted, n_proposed, n_accepted = [], 0, 0
    n_base, D = base_norm.shape
    while n_accepted < n_target and n_proposed < max_proposals:
        z = base_norm[rng.integers(0, n_base, size=batch)]
        z = z + sigma * rng.standard_normal((batch, D))
        raw = scaler.inverse_transform(z)
        keep = ((detector.score(z) <= tau)
                & is_healthy(raw, lo, hi)
                & reconstructable(raw, vel_start))
        accepted.append(raw[keep])
        n_proposed += batch
        n_accepted += int(keep.sum())
    pts = np.concatenate(accepted) if accepted else np.empty((0, D))
    return pts[:n_target], n_proposed, n_accepted


def healthy_data_box(env, data_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim sampling box for Mode A: the healthy constraint intersected with
    the per-dim safe-state range, so it is finite (the constraint is vacuous —
    +/-100 — on the joint-angle/velocity dims)."""
    lo, hi = healthy_bounds(env)
    return np.maximum(lo, data_states.min(0)), np.minimum(hi, data_states.max(0))


def uniform_sample(detector, scaler, box_lo, box_hi, tau, n_target, rng,
                   batch=16384, max_proposals=20_000_000):
    """Mode A: uniform proposals in the box, accept those in the set (cd <= tau).
    The box is a subset of the healthy region, so accepted points are healthy by
    construction. Returns accepted raw starts, n_proposed, n_accepted."""
    accepted, n_proposed, n_accepted = [], 0, 0
    while n_accepted < n_target and n_proposed < max_proposals:
        prop = rng.uniform(box_lo, box_hi, size=(batch, len(box_lo)))
        keep = detector.score(scaler.transform(prop)) <= tau
        accepted.append(prop[keep])
        n_proposed += batch
        n_accepted += int(keep.sum())
        if n_proposed % (batch * 32) == 0:
            print(f"\r    proposed {n_proposed:>12,}  accepted {n_accepted:>4}",
                  end="", flush=True)
    pts = np.concatenate(accepted) if accepted else np.empty((0, len(box_lo)))
    return pts[:n_target], n_proposed, n_accepted


def rollout(env, policy, start_obs, vel_start, horizon, seed):
    """Set Hopper to `start_obs`, roll out the policy. Returns (safe, ttf):
    safe is True iff it never terminates within `horizon`; ttf is the number of
    steps taken before termination (or `horizon` if it survives).

    Obs excludes the x-position (translation-invariant), so qpos[0]=0,
    qpos[1:] = obs[:vel_start], qvel = obs[vel_start:]. Termination == leaving
    the healthy box.
    """
    env.reset(seed=seed)
    u = env.unwrapped
    u.set_state(np.concatenate([[0.0], start_obs[:vel_start]]), start_obs[vel_start:])
    obs = u._get_obs()
    for t in range(horizon):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated:
            return False, t + 1
        if truncated:
            break
    return True, horizon


def run_cell(detector, tau, sigma, *, scaler, base_norm, lo, hi, vel_start,
             n_traj, seed, env, policy, ep_len):
    """Sample on-manifold starts at perturbation scale `sigma`, roll them out.
    Returns (summary row, starts, safe-flags, time-to-failure)."""
    rng = np.random.default_rng([seed, int(round(sigma * 1e6))])
    starts, n_proposed, n_accepted = perturb_sample(
        detector, scaler, base_norm, sigma, tau, lo, hi, vel_start, n_traj, rng)
    acc_rate = n_accepted / n_proposed

    safe = np.zeros(len(starts), dtype=bool)
    ttf = np.zeros(len(starts), dtype=int)
    for i, s in enumerate(starts):
        print(f"\r    sigma={sigma:<4g} rollout {i}/{len(starts)}", end="", flush=True)
        safe[i], ttf[i] = rollout(env, policy, s, vel_start, ep_len, seed + i)

    n = len(starts)
    prop_safe = float(safe.mean()) if n else float('nan')
    fail_ttf = ttf[~safe]
    med_ttf = float(np.median(fail_ttf)) if len(fail_ttf) else float('nan')
    n_early = int((~safe & (ttf < EARLY_K)).sum())
    print(f"\r    sigma={sigma:<4g}  accept {100 * acc_rate:5.1f}%  n={n:<3d}  "
          f"safe {100 * prop_safe:5.1f}%  holes {100 * (1 - prop_safe):5.1f}%  "
          f"medTTF={med_ttf:.0f}  early<{EARLY_K}={n_early}" + " " * 6)
    row = dict(sigma=sigma, acc_rate=acc_rate, n_rollout=n, prop_safe=prop_safe,
               med_ttf=med_ttf, n_early=n_early)
    return row, starts, safe, ttf


def run_uniform(detector, tau, degree, *, scaler, box_lo, box_hi, vel_start,
                n_traj, seed, env, policy, ep_len, max_proposals):
    """Mode A worker: uniform-box rejection sampling at one degree, then roll
    out. Returns (summary row, starts, safe-flags, time-to-failure)."""
    rng = np.random.default_rng([seed, degree])
    starts, n_proposed, n_accepted = uniform_sample(
        detector, scaler, box_lo, box_hi, tau, n_traj, rng, max_proposals=max_proposals)
    reject = 1.0 - n_accepted / n_proposed

    safe = np.zeros(len(starts), dtype=bool)
    ttf = np.zeros(len(starts), dtype=int)
    for i, s in enumerate(starts):
        print(f"\r    degree {degree} rollout {i}/{len(starts)}", end="", flush=True)
        safe[i], ttf[i] = rollout(env, policy, s, vel_start, ep_len, seed + i)

    n = len(starts)
    prop_safe = float(safe.mean()) if n else float('nan')
    fail_ttf = ttf[~safe]
    med_ttf = float(np.median(fail_ttf)) if len(fail_ttf) else float('nan')
    n_early = int((~safe & (ttf < EARLY_K)).sum())
    print(f"\r    degree {degree}  tau={tau:.4g}  reject {100 * reject:.3f}%  "
          f"({n_accepted}/{n_proposed})  n={n}  safe {100 * prop_safe:.1f}%  "
          f"holes {100 * (1 - prop_safe):.1f}%  medTTF={med_ttf:.0f}  early={n_early}"
          + " " * 4)
    if n < n_traj:
        print(f"    WARNING: only collected {n}/{n_traj} in-set starts")
    row = dict(degree=degree, tau=tau, reject=reject, n_proposed=n_proposed,
               n_accepted=n_accepted, n_rollout=n, prop_safe=prop_safe,
               med_ttf=med_ttf, n_early=n_early)
    return row, starts, safe, ttf


def main(
    env: EnvName = 'hopper',
    mode: Literal['manifold', 'uniform'] = 'manifold',
    n_traj: int = 100,
    degrees: list[int] = DEFAULT_DEGREES,
    sigmas: list[float] = DEFAULT_SIGMAS,
    alpha: float = 0.1,
    cal_frac: float = 0.5,
    max_fit_states: int = 50_000,
    max_proposals: int = 20_000_000,
    seed: int = 0,
):
    """Probe the CD-polynomial safety set for coverage holes (Hopper, base dataset).

    The deployment pipeline uses a Christoffel-Darboux polynomial to build a set
    that, by conformal calibration, contains ~90% of safe trajectories. That is a
    recall guarantee on the safe class; it says nothing about whether a point
    inside the set is actually safe. This script measures that precision gap by
    fitting a CD polynomial on pooled safe states, calibrating a per-trajectory
    max-conformal threshold, sampling candidate starts (on-manifold perturbations
    or uniform-box rejection), rolling them out under the SAC expert, and
    reporting the fraction that stay safe and the excess hole-rate over baseline.

    Args:
        env: Environment to probe.
        mode: Sampler: 'manifold' perturbs real safe states (sweeps sigma); 'uniform' is Mode A, uniform-box rejection (sweeps degree).
        n_traj: In-set starts to roll out per (degree, sigma) cell.
        degrees: CD polynomial degree(s) to sweep.
        sigmas: Perturbation scales (normalized std units); include 0 for the baseline.
        alpha: Target miss rate (1-coverage).
        cal_frac: Fraction of safe episodes held out for calibration.
        max_fit_states: Subsample cap on states used to fit the CD moment matrix.
        max_proposals: Mode A proposal budget per degree before giving up on n-traj.
        seed: Random seed.
    """
    cfg = DATASETS[f"{env}/base"]
    rng = np.random.default_rng(seed)

    # ── Data: surviving base episodes, split into fit / calibration ──────────
    ds = load(env, 'base')
    safe = survived(ds)
    n_safe = len(safe)
    perm = rng.permutation(n_safe)
    n_cal = int(round(n_safe * cal_frac))
    cal_eps = safe.X[perm[:n_cal]]
    fit_eps = safe.X[perm[n_cal:]]
    T = ds.X.shape[1]

    fit_states = pooled_states(fit_eps)
    if len(fit_states) > max_fit_states:
        fit_states = fit_states[rng.choice(len(fit_states), max_fit_states, replace=False)]

    print(f"\n{'#' * 64}\n# {env}/base — CD safety-set coverage probe ({mode})\n{'#' * 64}")
    print(f"  Safe episodes: {n_safe}  (fit {n_safe - n_cal} / cal {n_cal})")
    print(f"  Fit states: {len(fit_states)}  degrees={degrees}  alpha={alpha}")

    # ── Channel z-score (per-dim), fit on the fit-split states. Same transform
    #    as data.processing.normalize_channels, kept as an object so points can
    #    be mapped back to raw obs (inverse_transform). Degree-independent, so
    #    transform once and reuse. ──────────────────────────────────────────────
    scaler = StandardScaler().fit(fit_states)
    fit_norm = scaler.transform(fit_states)
    cal_norm = scaler.transform(cal_eps.reshape(-1, cal_eps.shape[-1]))

    sim = gym.make(ENV_INFO[env].gym_name)
    lo, hi = healthy_bounds(sim)
    vel_start = lo.shape[0] - sim.unwrapped.data.qvel.shape[0]

    if mode == 'manifold':
        # Base pool for perturbation: held-out safe states, restricted to the
        # faithfully-reconstructable (non-velocity-clipped) region.
        base_raw = pooled_states(cal_eps)
        n_pre = len(base_raw)
        base_raw = base_raw[reconstructable(base_raw, vel_start)]
        base_norm = scaler.transform(base_raw)
        print(f"  sigmas={sigmas}")
        print(f"  Base pool: {len(base_raw)}/{n_pre} cal states reconstructable "
              f"(dropped {100 * (1 - len(base_raw) / n_pre):.1f}% velocity-clipped)")
    else:
        box_lo, box_hi = healthy_data_box(sim, pooled_states(safe.X))
        np.set_printoptions(precision=3, suppress=True)
        print(f"  Sampling box low:  {box_lo}")
        print(f"  Sampling box high: {box_hi}")

    policy = SAC.load(str(get_root() / 'data/policies' / cfg.policy), env=sim)
    out_dir = get_output_dir()

    rows, saved = [], {}
    for degree in degrees:
        print(f"\n  degree {degree}\n  {'─' * 56}")
        detector = CDPolyDetector(degree=degree, basis='cheb', method='qr')
        detector.fit(fit_norm)
        tau = calibrate_threshold(detector, cal_norm, n_cal, T, alpha)
        print(f"    tau = {tau:.4g}")
        if mode == 'manifold':
            for sigma in sigmas:
                row, starts, flags, ttf = run_cell(
                    detector, tau, sigma, scaler=scaler, base_norm=base_norm,
                    lo=lo, hi=hi, vel_start=vel_start, n_traj=n_traj,
                    seed=seed, env=sim, policy=policy, ep_len=cfg.ep_len)
                row.update(degree=degree, tau=tau)
                rows.append(row)
                saved[f"starts_d{degree}_s{sigma}"] = starts
                saved[f"safe_d{degree}_s{sigma}"] = flags
                saved[f"ttf_d{degree}_s{sigma}"] = ttf
        else:
            row, starts, flags, ttf = run_uniform(
                detector, tau, degree, scaler=scaler, box_lo=box_lo, box_hi=box_hi,
                vel_start=vel_start, n_traj=n_traj, seed=seed,
                env=sim, policy=policy, ep_len=cfg.ep_len, max_proposals=max_proposals)
            rows.append(row)
            saved[f"starts_d{degree}"] = starts
            saved[f"safe_d{degree}"] = flags
            saved[f"ttf_d{degree}"] = ttf
    sim.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    if mode == 'manifold':
        baseline = {r['degree']: 1 - r['prop_safe'] for r in rows if r['sigma'] == 0.0}
        print(f"\n{'=' * 76}")
        print(f"  {'deg':>3}  {'sigma':>6}  {'accept%':>8}  {'n':>4}  {'safe%':>6}  "
              f"{'holes%':>7}  {'excess_pp':>9}  {'medTTF':>7}  {'early':>5}")
        print(f"  {'─' * 68}")
        for r in rows:
            holes = 1 - r['prop_safe']
            base = baseline.get(r['degree'])
            excess = f"{100 * (holes - base):>+8.1f}" if base is not None else f"{'—':>9}"
            print(f"  {r['degree']:>3}  {r['sigma']:>6g}  {100 * r['acc_rate']:>7.1f}%  "
                  f"{r['n_rollout']:>4}  {100 * r['prop_safe']:>5.1f}%  {100 * holes:>6.1f}%  "
                  f"{excess}  {r['med_ttf']:>7.0f}  {r['n_early']:>5}")
        print(f"  {'─' * 68}")
        print(f"  excess_pp = holes(sigma) - holes(sigma=0); early = failures within {EARLY_K} steps")
    else:
        print(f"\n{'=' * 76}")
        print(f"  {'deg':>3}  {'tau':>10}  {'reject%':>9}  {'n':>4}  {'safe%':>6}  "
              f"{'holes%':>7}  {'medTTF':>7}  {'early':>5}")
        print(f"  {'─' * 64}")
        for r in rows:
            holes = 1 - r['prop_safe']
            print(f"  {r['degree']:>3}  {r['tau']:>10.4g}  {100 * r['reject']:>8.3f}%  "
                  f"{r['n_rollout']:>4}  {100 * r['prop_safe']:>5.1f}%  {100 * holes:>6.1f}%  "
                  f"{r['med_ttf']:>7.0f}  {r['n_early']:>5}")
        print(f"  {'─' * 64}")
        print(f"  early = failures within {EARLY_K} steps")

    out = out_dir / f"{env}_{mode}.npz"
    np.savez(
        out,
        env=env, mode=mode, alpha=alpha, seed=seed, early_k=EARLY_K,
        degree_col=np.array([r['degree'] for r in rows]),
        tau_col=np.array([r['tau'] for r in rows]),
        n_rollout_col=np.array([r['n_rollout'] for r in rows]),
        prop_safe_col=np.array([r['prop_safe'] for r in rows]),
        med_ttf_col=np.array([r['med_ttf'] for r in rows]),
        n_early_col=np.array([r['n_early'] for r in rows]),
        **saved,
    )
    print(f"  Saved to {out}\n")


if __name__ == "__main__":
    tyro.cli(main)
