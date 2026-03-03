#!/usr/bin/env python3
"""
Score Curves + Rendered Frames for Figure 1.

Trains Basis-CD on fold-0 training successes from the stored dataset, then
re-simulates selected test episodes with MuJoCo rendering to produce:
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
from config.detectors import get_detector, get_method_display_name
from config.envs import ENV_INFO
from config.tasks import TASK_CONFIGS
from envs.mujoco.termination import check_custom_termination
from tasks.fold_task import create_fold_tasks
from utils.paths import get_root
from utils.windows import strided_window_view

N_EPISODES = 5

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


def reconstruct_episode_params(cfg):
    """Reconstruct per-episode physical parameters from dataset config seed.

    Replicates the RNG logic from gen_data.py so we can re-simulate episodes
    with matching physical parameters.

    Returns:
        seeds, mass_scale, friction_scale, damping_scale arrays
    """
    rng = np.random.default_rng(cfg.seed)
    seeds = rng.integers(0, 2**32, size=cfg.n_episodes, dtype=np.uint32)
    mass_scale = rng.uniform(
        cfg.mass_range[0], cfg.mass_range[1], size=cfg.n_episodes
    ).astype(np.float32)
    friction_scale = rng.uniform(
        cfg.friction_range[0], cfg.friction_range[1], size=cfg.n_episodes
    ).astype(np.float32)
    damping_scale = rng.uniform(
        cfg.damping_range[0], cfg.damping_range[1], size=cfg.n_episodes
    ).astype(np.float32)
    return seeds, mass_scale, friction_scale, damping_scale


def simulate_episode(env, policy, renderer, cam, gym_name, ep_len,
                     seed, mass_scale, friction_scale, damping_scale,
                     base_mass, base_fric, base_damp):
    """Re-simulate a single episode with rendering.

    Returns:
        obs_buffer: (actual_len, obs_dim) observations collected
        frames: list of RGB arrays, one per timestep
        fail_step: int, -1 if success, >=0 if failure
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

    for step in range(ep_len):
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
            for _post_step in range(step + 1, ep_len):
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
            frame_size: int, save_every: int):
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

    # ------- Step 2: Select test episodes -------
    test_fail = task.fail[task.test_idx]
    test_success_idx = task.test_idx[test_fail == -1]
    test_failure_idx = task.test_idx[test_fail >= 0]

    rng = np.random.RandomState(seed)
    sel_success = rng.choice(
        test_success_idx,
        size=min(n_episodes, len(test_success_idx)),
        replace=False,
    )
    sel_failure = rng.choice(
        test_failure_idx,
        size=min(n_episodes, len(test_failure_idx)),
        replace=False,
    )
    sel_all = np.concatenate([sel_success, sel_failure])

    ep_len = task.X.shape[1]
    print(f"Environment: {env_name} | ep_len={ep_len} | "
          f"{len(sel_success)} success + {len(sel_failure)} failure episodes")

    # ------- Step 3: Reconstruct episode params -------
    seeds, mass_scale, friction_scale, damping_scale = \
        reconstruct_episode_params(dataset_cfg)

    # ------- Step 4: Set up MuJoCo env + renderer -------
    env = gym.make(gym_name)
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

    # ------- Step 5: Simulate, score, and save -------
    output_dir = get_root() / "results" / "fig1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect score curves for plotting
    curve_data = []  # list of (timesteps, scores, fail_step, is_success, label)

    # Metadata for reproducibility
    meta_timesteps = []
    meta_scores = []
    meta_fail_steps = []
    meta_episode_indices = []

    for i, ep_idx in enumerate(sel_all):
        is_success = ep_idx in sel_success
        kind = "success" if is_success else "failure"
        label = f"ep{i:02d}_{kind}"

        print(f"  Simulating {label} (dataset ep {ep_idx}, "
              f"m={mass_scale[ep_idx]:.3f}, f={friction_scale[ep_idx]:.3f}, "
              f"d={damping_scale[ep_idx]:.3f})...")

        obs_buf, frames, fail_step = simulate_episode(
            env, policy, renderer, cam, gym_name, ep_len,
            seeds[ep_idx], mass_scale[ep_idx],
            friction_scale[ep_idx], damping_scale[ep_idx],
            base_mass, base_fric, base_damp,
        )

        # Normalize and score
        obs_norm = scaler.transform(obs_buf)
        ts, scores = score_episode(detector, obs_norm)

        actual_fail = fail_step if not is_success else -1
        print(f"    {len(frames)} steps, fail_step={actual_fail}, "
              f"max_score={scores.max():.3g}")

        # Save frames
        save_frames(frames, output_dir, label, save_every=save_every)

        curve_data.append((ts, scores, actual_fail, is_success, label))
        meta_timesteps.append(ts)
        meta_scores.append(scores)
        meta_fail_steps.append(actual_fail)
        meta_episode_indices.append(ep_idx)

    renderer.close()
    env.close()

    # ------- Step 6: Plot score curves -------
    display_name = get_method_display_name("basis")
    fig, ax = plt.subplots(figsize=(10, 3))

    for ts, scores, fail_step, is_success, _label in curve_data:
        color = "green" if is_success else "red"
        ax.plot(ts, scores, color=color, alpha=0.6, linewidth=0.8)
        if not is_success and fail_step >= 0:
            ax.plot(fail_step, np.interp(fail_step, ts, scores),
                    "rx", markersize=8, markeredgewidth=2)

    ax.axhline(detector.threshold_, color="gray", linestyle="--",
               linewidth=1, label="threshold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Score")
    ax.set_title(f"{display_name} — {env_info.display_name}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    svg_path = output_dir / "score_curves.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved score curves to {svg_path}")

    # ------- Step 7: Save metadata -------
    meta_path = output_dir / "metadata.npz"
    np.savez(
        meta_path,
        episode_indices=np.array(meta_episode_indices),
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
            args.frame_size, args.save_every)


if __name__ == "__main__":
    main()
