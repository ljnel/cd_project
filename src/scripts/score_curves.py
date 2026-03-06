#!/usr/bin/env python3
"""
Score Curves + Rendered Frames for Figure 1.

Trains Basis-CD on fold-0 training successes from the stored dataset, then
samples fresh episodes with random parameters from the dataset config's domain
randomization ranges to produce:
  - Per-timestep anomaly score curves (SVG)
  - High-res rendered frames (PNGs) for every timestep

Usage:
    python -m scripts.score_curves --env humanoid --n-episodes 5
    python -m scripts.score_curves --env humanoid --n-episodes 2 --save-every 5
"""

import argparse
import warnings

import matplotlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from config.datasets import DATASETS
from config.detectors import get_detector
from config.envs import ENV_INFO
from config.tasks import TASK_CONFIGS
from envs.mujoco.termination import check_custom_termination
from tasks.fold_task import create_fold_tasks
from utils.paths import get_root
from utils.plotting import FAILURE_COLOR, FULL_WIDTH, SUCCESS_COLOR, setup_style
from utils.windows import strided_window_view

setup_style()

N_EPISODES = 10

# Per-env camera settings (distance, elevation, azimuth)
CAMERA_SETTINGS = {
    'inv_pend': (3.0, -20.0, 135.0),
    'hopper': (3.0, -20.0, 135.0),
    'half_cheetah': (4.0, -25.0, 135.0),
    'ant': (5.0, -25.0, 135.0),
    'humanoid': (5.0, -20.0, 90.0),
}


def score_episode(detector, episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slide detector window over a single episode at stride=1.

    Args:
        detector: Fitted AnomalyDetector with .window and .score_samples()
        episode: (ep_len, n_features)

    Returns:
        timesteps: 1-D array of timestep indices where each window ends
        scores: corresponding anomaly scores
    """
    ep = episode[np.newaxis]  # (1, ep_len, n_features)
    windows = strided_window_view(ep, detector.window, stride=1)  # (1, n_win, win, feat)
    windows = windows[0]  # (n_win, win, feat)
    scores = detector.score_samples(windows)
    timesteps = np.arange(detector.window - 1, detector.window - 1 + len(scores))
    return timesteps, scores


def simulate_episode(env, policy, renderer, cam, gym_name, sim_len,
                     seed, mass_scale, friction_scale, damping_scale,
                     base_mass, base_fric, base_damp):
    """Re-simulate a single episode with rendering.

    Parameters
    ----------
    sim_len : int
        Total steps to simulate (typically ep_len + hor to detect late failures).

    Returns:
        obs_buffer: (actual_len, obs_dim) observations collected before termination
        frames: list of RGB arrays, one per simulated timestep
        fail_step: int, -1 if survived all sim_len steps, >=0 if failure
    """
    import mujoco

    m = env.unwrapped.model
    mj_data = env.unwrapped.data

    # Apply parameter variations
    m.body_mass[:] = base_mass * mass_scale
    m.geom_friction[:] = base_fric * friction_scale
    m.dof_damping[:] = base_damp * damping_scale

    obs, _ = env.reset(seed=int(seed))
    obs_list = []
    frames = []
    fail_step = -1

    for step in range(sim_len):
        obs_list.append(obs)

        # Render frame (track body 1)
        cam.lookat[:] = mj_data.xpos[1]
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render().copy())

        action, _ = policy.predict(obs, deterministic=True)
        next_obs, _, terminated, truncated, _ = env.step(action)

        if check_custom_termination(gym_name, next_obs):
            terminated = True

        if terminated:
            fail_step = step
        if terminated or truncated:
            # Keep rendering the aftermath (zero torques, physics only)
            mj_data.ctrl[:] = 0
            for _post_step in range(step + 1, sim_len):
                mujoco.mj_step(m, mj_data)
                cam.lookat[:] = mj_data.xpos[1]
                renderer.update_scene(mj_data, camera=cam)
                frames.append(renderer.render().copy())
            break

        obs = next_obs

    obs_buffer = np.array(obs_list, dtype=np.float32)
    return obs_buffer, frames, fail_step


def save_frames(frames, output_dir, episode_label, save_every=1):
    """Save rendered frames as PNGs.

    Args:
        frames: list of RGB arrays
        output_dir: base output directory
        episode_label: e.g. "ep00_success"
        save_every: save every N-th frame (1 = all frames)
    """
    ep_dir = output_dir / episode_label
    ep_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        if i % save_every != 0:
            continue
        img = Image.fromarray(frame)
        img.save(ep_dir / f"t{i:04d}.png")


def run_env(env_name: str, n_episodes: int, seed: int,
            frame_size: int, save_every: int, log_scale: bool = False):
    """Generate score curves + rendered frames for Figure 1."""
    import gymnasium as gym
    import mujoco

    from envs.mujoco.gen_data import _load_policy

    task_cfg = TASK_CONFIGS[env_name]
    dataset_key = f"{env_name}/fail_pred"
    dataset_cfg = DATASETS[dataset_key]
    env_info = ENV_INFO[env_name]
    gym_name = env_info.gym_name

    np.random.seed(seed)

    # ------- Step 1: Load data, fit scaler & detector on fold 0 -------
    tasks = create_fold_tasks(task_cfg, n_folds=5, seed=seed)
    task = tasks[0]

    success_mask = task.fail == -1
    train_success_idx = task.train_idx[success_mask[task.train_idx]]
    x_train_raw = task.X[train_success_idx]

    scaler = StandardScaler()
    flat = x_train_raw.reshape(-1, x_train_raw.shape[-1])
    scaler.fit(flat)
    x_train = scaler.transform(flat).reshape(x_train_raw.shape)

    print("Fitting Basis-CD...")
    detector = get_detector("basis", env=env_name)
    detector.fit(x_train)
    print(f"  window={detector.window}, threshold={detector.threshold_:.3g}")

    win = task_cfg.win
    ep_len = dataset_cfg.ep_len  # original (untrimmed) episode length

    # ------- Step 2: Set up MuJoCo env + renderer -------
    hor = task_cfg.hor
    sim_len = ep_len + hor  # extend beyond dataset to detect late failures
    env = gym.make(gym_name, max_episode_steps=sim_len)
    policy_path = str(get_root() / 'src/policies' / dataset_cfg.policy)
    policy = _load_policy(dataset_cfg.algo, policy_path, env)

    mj_model = env.unwrapped.model
    base_mass = mj_model.body_mass.copy()
    base_fric = mj_model.geom_friction.copy()
    base_damp = mj_model.dof_damping.copy()

    # Extend ground plane so humanoid doesn't walk off it
    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            mj_model.geom_size[i, 0] = 100.0
            mj_model.geom_size[i, 1] = 100.0
    # White background: haze fades horizon to white, fog fades distant objects
    mj_model.vis.rgba.haze[:] = [1, 1, 1, 1]
    mj_model.vis.rgba.fog[:] = [1, 1, 1, 1]
    mj_model.vis.map.fogstart = 10.0
    mj_model.vis.map.fogend = 20.0

    renderer = mujoco.Renderer(mj_model, height=frame_size, width=frame_size)
    cam = mujoco.MjvCamera()
    cam.type = 0  # free camera
    dist, elev, azim = CAMERA_SETTINGS.get(env_name, (4.0, -25.0, 135.0))
    cam.distance = dist
    cam.elevation = elev
    cam.azimuth = azim

    # ------- Step 3: Output dir -------
    import shutil
    output_dir = get_root() / "results" / "fig1"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------- Step 4: Sample fresh episodes -------
    sample_rng = np.random.default_rng(seed)
    max_attempts = 10 * n_episodes
    successes = []  # list of (obs_buf, frames, fail_step)
    failures = []

    print(f"Environment: {env_name} | ep_len={ep_len} | win={win}")
    print(f"Sampling episodes (target: {n_episodes} success + "
          f"{n_episodes} failure, max {max_attempts} attempts)...")

    attempt = 0
    for attempt in range(max_attempts):
        if len(successes) >= n_episodes and len(failures) >= n_episodes:
            break

        mass = float(sample_rng.uniform(*dataset_cfg.mass_range))
        friction = float(sample_rng.uniform(*dataset_cfg.friction_range))
        damping = float(sample_rng.uniform(*dataset_cfg.damping_range))
        ep_seed = int(sample_rng.integers(0, 2**32))

        obs_buf, frames, fail_step = simulate_episode(
            env, policy, renderer, cam, gym_name, sim_len,
            ep_seed, mass, friction, damping,
            base_mass, base_fric, base_damp,
        )

        is_success = fail_step == -1
        if is_success and len(successes) < n_episodes:
            successes.append((obs_buf, frames, fail_step))
            print(f"  attempt {attempt}: success "
                  f"({len(successes)}/{n_episodes})")
        elif not is_success and len(failures) < n_episodes:
            failures.append((obs_buf, frames, fail_step))
            print(f"  attempt {attempt}: failure "
                  f"({len(failures)}/{n_episodes})")

    print(f"Collected {len(successes)} successes + {len(failures)} failures "
          f"in {attempt + 1} attempts")

    # ------- Step 5: Score and save -------
    curve_data = []  # list of (timesteps, scores, fail_step, is_success, label)
    meta_scores = []
    meta_fail_steps = []

    all_episodes = successes + failures
    for i, (obs_buf, frames, fail_step) in enumerate(all_episodes):
        is_success = fail_step == -1
        kind = "success" if is_success else "failure"
        label = f"ep{i:02d}_{kind}"

        # Score only the original episode portion (not the extended horizon)
        obs_score = obs_buf[win:ep_len]
        obs_norm = scaler.transform(obs_score)
        ts, scores = score_episode(detector, obs_norm)
        ts += win  # offset to original simulation time

        print(f"    {label} | {len(frames)} steps, fail_step={fail_step}, "
              f"max_score={scores.max():.3g}")

        save_frames(frames, output_dir, label, save_every=save_every)

        curve_data.append((ts, scores, fail_step, is_success, label))
        meta_scores.append(scores)
        meta_fail_steps.append(fail_step)

    renderer.close()
    env.close()

    # ------- Step 6: Plot score curves -------
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.2))

    first_crossing = np.inf
    last_fail = -1
    for ts, scores, fail_step, is_success, _label in curve_data:
        cross_mask = scores > detector.threshold_
        if cross_mask.any():
            cross_idx = int(np.argmax(cross_mask))
            first_crossing = min(first_crossing, ts[cross_idx])
            # Green segment (up to and including crossing point)
            ax.plot(ts[:cross_idx + 1], scores[:cross_idx + 1],
                    color=SUCCESS_COLOR, alpha=0.6)
            # Red segment (from crossing point onward)
            ax.plot(ts[cross_idx:], scores[cross_idx:],
                    color=FAILURE_COLOR, alpha=0.6)
        else:
            ax.plot(ts, scores, color=SUCCESS_COLOR, alpha=0.6)
        if not is_success and fail_step >= 0:
            ax.plot(fail_step, np.interp(fail_step, ts, scores),
                    "x", color=FAILURE_COLOR, markersize=6, markeredgewidth=1.5)
            last_fail = max(last_fail, fail_step)

    ax.axhline(detector.threshold_, color="gray", linestyle="--",
               label="threshold")
    if first_crossing < np.inf:
        ax.set_xlim(left=first_crossing - 100)
    if last_fail >= 0:
        ax.set_xlim(right=last_fail + 100)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left")
    fig.tight_layout()

    svg_path = output_dir / "score_curves.svg"
    pdf_path = output_dir / "score_curves.pdf"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved score curves to {svg_path} and {pdf_path}")

    # ------- Step 7: Save metadata -------
    meta_path = output_dir / "metadata.npz"
    np.savez(
        meta_path,
        fail_steps=np.array(meta_fail_steps),
        threshold=detector.threshold_,
        window=detector.window,
        # Variable-length arrays stored as object arrays
        **{f"ts_{i}": ts for i, (ts, _, _, _, _) in enumerate(curve_data)},
        **{f"scores_{i}": sc for i, (_, sc, _, _, _) in enumerate(curve_data)},
    )
    print(f"Saved metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 1: Basis-CD score curves + rendered frames."
    )
    parser.add_argument(
        "--env", type=str, default="humanoid",
        help=f"Environment name (default: humanoid). Available: {list(TASK_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=N_EPISODES,
        help="Number of success/failure episodes to simulate (default: 5)",
    )
    parser.add_argument(
        "--frame-size", type=int, default=480,
        help="Render resolution in pixels (default: 480)",
    )
    parser.add_argument(
        "--save-every", type=int, default=1,
        help="Save every N-th frame (default: 1 = all frames)",
    )
    parser.add_argument("--log", action="store_true", help="Use log scale for y-axis")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    if args.env not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown env: {args.env}. Available: {list(TASK_CONFIGS.keys())}"
        )

    dataset_key = f"{args.env}/fail_pred"
    if dataset_key not in DATASETS:
        raise ValueError(
            f"No fail_pred dataset for {args.env}. "
            f"Available: {[k for k in DATASETS if k.endswith('/fail_pred')]}"
        )

    run_env(args.env, args.n_episodes, args.seed,
            args.frame_size, args.save_every, args.log)


if __name__ == "__main__":
    main()
