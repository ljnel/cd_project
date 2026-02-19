"""
Visual exploration of failure modes across environments.

Generates fresh episodes with rendering enabled, captures frames around
failure events via a rolling buffer, and composes (n_failures, 2) figure
grids showing "before" and "at failure" frames.

Usage:
    python -m scripts.explore_failures [--steps_before 10] [--n_failures 5] [--envs hopper ant ...]
"""

import logging
import warnings
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.datasets import DATASETS, DatasetConfig
from config.envs import ENV_INFO
from config.tasks import TASK_CONFIGS
from envs.mujoco.termination import check_custom_termination
from utils.paths import get_root

logger = logging.getLogger("cd.scripts.explore_failures")

MAX_EPISODES = 200  # give up after this many episodes per env

# Per-env camera settings (distance, elevation, azimuth)
CAMERA_SETTINGS = {
    'inv_pend': (3.0, -20.0, 135.0),
    'hopper': (3.0, -20.0, 135.0),
    'half_cheetah': (4.0, -25.0, 135.0),
    'ant': (5.0, -25.0, 135.0),
    'humanoid': (5.0, -20.0, 135.0),
}


def run_mujoco_failures(cfg: DatasetConfig, n_failures: int, steps_before: int,
                        seed: int) -> list:
    """Generate MuJoCo episodes until n_failures collected.

    Returns list of (before_frame, fail_frame, fail_step) tuples.
    """
    import gymnasium as gym
    import mujoco
    from stable_baselines3 import SAC

    env_info = ENV_INFO[cfg.env]
    gym_name = env_info.gym_name
    policy_path = str(get_root() / 'src/policies' / cfg.policy)

    env = gym.make(gym_name)
    policy = SAC.load(policy_path, env=env)

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data
    base_mass = mj_model.body_mass.copy()
    base_fric = mj_model.geom_friction.copy()
    base_damp = mj_model.dof_damping.copy()

    # Use mujoco.Renderer directly for reliable camera tracking
    renderer = mujoco.Renderer(mj_model, height=480, width=480)
    cam = mujoco.MjvCamera()
    cam.type = 0  # free camera
    dist, elev, azim = CAMERA_SETTINGS.get(cfg.env, (4.0, -25.0, 135.0))
    cam.distance = dist
    cam.elevation = elev
    cam.azimuth = azim

    def render_frame():
        cam.lookat[:] = mj_data.xpos[1]
        renderer.update_scene(mj_data, camera=cam)
        return renderer.render()

    rng = np.random.default_rng(seed)
    results = []

    for ep in range(MAX_EPISODES):
        if len(results) >= n_failures:
            break

        # Sample parameters
        mass_scale = rng.uniform(cfg.mass_range[0], cfg.mass_range[1])
        fric_scale = rng.uniform(cfg.friction_range[0], cfg.friction_range[1])
        damp_scale = rng.uniform(cfg.damping_range[0], cfg.damping_range[1])
        ep_seed = int(rng.integers(0, 2**32))

        mj_model.body_mass[:] = base_mass * mass_scale
        mj_model.geom_friction[:] = base_fric * fric_scale
        mj_model.dof_damping[:] = base_damp * damp_scale

        obs, _ = env.reset(seed=ep_seed)
        frame_buffer = deque(maxlen=steps_before + 1)

        failed = False
        for step in range(cfg.ep_len):
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, _ = env.step(action)
            frame_buffer.append(render_frame())

            if check_custom_termination(gym_name, next_obs):
                terminated = True

            if terminated:
                results.append((frame_buffer[0], frame_buffer[-1], step))
                failed = True
                break

            if truncated:
                break

            obs = next_obs

        if not failed:
            continue

        logger.info(f"  [{cfg.env}] Failure {len(results)}/{n_failures} at step {step} "
                    f"(ep {ep}, m={mass_scale:.2f}, f={fric_scale:.2f}, d={damp_scale:.2f})")

    renderer.close()
    env.close()
    return results


def render_pybullet_frame(simulator) -> np.ndarray:
    """Capture an RGB frame from PyBullet headless renderer."""
    import pybullet as p

    width, height = 480, 360
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.4],
        distance=1.2,
        yaw=45,
        pitch=-20,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=10.0,
    )
    _, _, rgb, _, _ = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
            if hasattr(p, 'ER_BULLET_HARDWARE_OPENGL') else p.ER_TINY_RENDERER,
    )
    return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def _upkie_find_failures(
    episode_indices: list,
    episode_seeds: np.ndarray,
    mass_scales: np.ndarray,
    friction_scales: np.ndarray,
    damping_scales: np.ndarray,
    cfg_frequency: float,
    cfg_balancer: str,
    policy_path: str,
    disturbance_type: str,
    disturbance_kwargs: dict,
    n_steps: int,
) -> list:
    """Phase 1 worker: run episodes WITHOUT rendering, return fail info.

    Returns list of (episode_index, fail_step) for episodes that failed.
    """
    import gymnasium as gym
    import upkie.envs

    from envs.upkie.disturbances import clear_external_forces
    from envs.upkie.gen_data import _apply_parameter_scales, _get_base_parameters, _restore_parameters

    upkie.envs.register()

    _OBS_DIM, _ACTION_DIM = 4, 1
    use_ppo = (cfg_balancer == 'ppo')

    base_env = gym.make("Upkie-PyBullet-Pendulum", frequency=cfg_frequency, gui=False)
    base_env.unwrapped.update_init_rand(pitch=0.02)
    simulator = base_env.unwrapped.backend
    robot_id = simulator.robot_id
    env = base_env
    base_params = _get_base_parameters(robot_id)

    if use_ppo:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path)
        def get_action(obs, info):
            return model.predict(obs, deterministic=True)[0]
    else:
        mpc = None
        def get_action(obs, info):
            return mpc.compute_ground_velocity(0.0, info["spine_observation"], base_env.unwrapped.dt)

    disturbance = None
    if disturbance_type is not None:
        from envs.upkie import disturbances as dist_mod
        disturbance = getattr(dist_mod, disturbance_type)(**disturbance_kwargs)

    rng = np.random.default_rng()
    failures = []

    for ep in episode_indices:
        obs, info = env.reset(seed=int(episode_seeds[ep]))
        clear_external_forces(simulator)

        if not use_ppo:
            from upkie.controllers import MPCBalancer
            mpc = MPCBalancer(fall_pitch=1.0, leg_length=0.58, max_ground_accel=10.0,
                              max_ground_velocity=3.0, nb_timesteps=50, sampling_period=0.02)

        _apply_parameter_scales(robot_id, mass_scales[ep], friction_scales[ep],
                                damping_scales[ep], base_params)
        if disturbance:
            disturbance.reset(n_steps, rng)

        for step in range(n_steps):
            action = np.atleast_1d(get_action(obs, info)).reshape(base_env.action_space.shape)
            if disturbance:
                _, action = disturbance.apply(step, obs.copy(), action.copy(), simulator)
            else:
                clear_external_forces(simulator)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                failures.append((ep, step))
                break

        clear_external_forces(simulator)
        _restore_parameters(robot_id, base_params)

    env.close()
    return failures


def _upkie_render_failure(
    ep_index: int,
    fail_step: int,
    episode_seeds: np.ndarray,
    mass_scales: np.ndarray,
    friction_scales: np.ndarray,
    damping_scales: np.ndarray,
    cfg_frequency: float,
    cfg_balancer: str,
    policy_path: str,
    disturbance_type: str,
    disturbance_kwargs: dict,
    steps_before: int,
    n_steps: int,
) -> tuple:
    """Phase 2 worker: re-run a single failing episode WITH rendering.

    Only renders frames in the window [fail_step - steps_before, fail_step].
    Returns (before_frame, fail_frame, fail_step).
    """
    import gymnasium as gym
    import upkie.envs

    from envs.upkie.disturbances import clear_external_forces
    from envs.upkie.gen_data import _apply_parameter_scales, _get_base_parameters

    upkie.envs.register()

    _OBS_DIM, _ACTION_DIM = 4, 1
    use_ppo = (cfg_balancer == 'ppo')

    base_env = gym.make("Upkie-PyBullet-Pendulum", frequency=cfg_frequency, gui=False)
    base_env.unwrapped.update_init_rand(pitch=0.02)
    simulator = base_env.unwrapped.backend
    robot_id = simulator.robot_id
    env = base_env
    base_params = _get_base_parameters(robot_id)

    if use_ppo:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path)
        def get_action(obs, info):
            return model.predict(obs, deterministic=True)[0]
    else:
        mpc = None
        def get_action(obs, info):
            return mpc.compute_ground_velocity(0.0, info["spine_observation"], base_env.unwrapped.dt)

    disturbance = None
    if disturbance_type is not None:
        from envs.upkie import disturbances as dist_mod
        disturbance = getattr(dist_mod, disturbance_type)(**disturbance_kwargs)

    rng = np.random.default_rng()

    obs, info = env.reset(seed=int(episode_seeds[ep_index]))
    clear_external_forces(simulator)

    if not use_ppo:
        from upkie.controllers import MPCBalancer
        mpc = MPCBalancer(fall_pitch=1.0, leg_length=0.58, max_ground_accel=10.0,
                          max_ground_velocity=3.0, nb_timesteps=50, sampling_period=0.02)

    _apply_parameter_scales(robot_id, mass_scales[ep_index], friction_scales[ep_index],
                            damping_scales[ep_index], base_params)
    if disturbance:
        disturbance.reset(n_steps, rng)

    render_start = max(0, fail_step - steps_before)
    frame_buffer = deque(maxlen=steps_before + 1)

    for step in range(fail_step + 1):
        action = np.atleast_1d(get_action(obs, info)).reshape(base_env.action_space.shape)
        if disturbance:
            _, action = disturbance.apply(step, obs.copy(), action.copy(), simulator)
        else:
            clear_external_forces(simulator)
        obs, _, term, trunc, info = env.step(action)

        if step >= render_start:
            frame_buffer.append(render_pybullet_frame(simulator))

    env.close()
    return (frame_buffer[0], frame_buffer[-1], fail_step)


def run_upkie_failures(cfg: DatasetConfig, n_failures: int, steps_before: int,
                       seed: int, n_jobs: int = -1) -> list:
    """Generate Upkie failures using two-phase approach:
    1) Find failures in parallel WITHOUT rendering (fast)
    2) Re-run only failing episodes WITH rendering (only around failure window)

    Returns list of (before_frame, fail_frame, fail_step) tuples.
    """
    try:
        import upkie.envs  # noqa: F401
    except ImportError as e:
        logger.warning(f"Skipping upkie: {e}")
        return []

    import os

    from joblib import Parallel, delayed

    # Pre-sample parameters for all episodes
    rng = np.random.default_rng(seed)
    n_steps = cfg.ep_len
    episode_seeds = rng.integers(0, 2**31, size=MAX_EPISODES, dtype=np.int64)
    mass_scales = rng.uniform(cfg.mass_range[0], cfg.mass_range[1], size=MAX_EPISODES).astype(np.float32)
    friction_scales = rng.uniform(cfg.friction_range[0], cfg.friction_range[1], size=MAX_EPISODES).astype(np.float32)
    damping_scales = rng.uniform(cfg.damping_range[0], cfg.damping_range[1], size=MAX_EPISODES).astype(np.float32)

    policy_path = str(get_root() / 'src/policies' / cfg.policy) if cfg.policy else None

    # Split episodes across workers
    n_workers = n_jobs if n_jobs > 0 else max(1, os.cpu_count() + n_jobs + 1)
    all_indices = list(range(MAX_EPISODES))
    batches = np.array_split(all_indices, n_workers)
    batches = [b.tolist() for b in batches if len(b) > 0]

    # Phase 1: find failures without rendering
    logger.info(f"  [{cfg.env}] Phase 1: finding failures ({len(batches)} workers)...")
    worker_results = Parallel(n_jobs=n_jobs)(
        delayed(_upkie_find_failures)(
            batch, episode_seeds, mass_scales, friction_scales, damping_scales,
            cfg.frequency, cfg.balancer, policy_path,
            cfg.disturbance_type, cfg.disturbance_kwargs, n_steps,
        )
        for batch in batches
    )

    # Collect and sort failures by episode index
    all_failures = []
    for res in worker_results:
        all_failures.extend(res)
    all_failures.sort(key=lambda x: x[0])
    all_failures = all_failures[:n_failures]

    if not all_failures:
        return []

    logger.info(f"  [{cfg.env}] Phase 2: rendering {len(all_failures)} failures...")

    # Phase 2: re-run only failing episodes with rendering (parallel)
    shared_args = dict(
        episode_seeds=episode_seeds, mass_scales=mass_scales,
        friction_scales=friction_scales, damping_scales=damping_scales,
        cfg_frequency=cfg.frequency, cfg_balancer=cfg.balancer,
        policy_path=policy_path, disturbance_type=cfg.disturbance_type,
        disturbance_kwargs=cfg.disturbance_kwargs,
        steps_before=steps_before, n_steps=n_steps,
    )
    results = Parallel(n_jobs=n_jobs)(
        delayed(_upkie_render_failure)(ep_idx, fail_step, **shared_args)
        for ep_idx, fail_step in all_failures
    )

    return results


def compose_figure(frames_data: list, display_name: str, steps_before: int,
                   output_path: Path):
    """Create (n_failures, 2) figure grid and save PNG + PDF.

    Args:
        frames_data: list of (before_frame, fail_frame, fail_step) tuples
        display_name: human-readable env name for title
        steps_before: number of steps before failure for column label
        output_path: path without extension (will save .pdf)
    """
    n = len(frames_data)
    if n == 0:
        logger.warning(f"No failures found for {display_name}, skipping figure.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(8, 2.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"{display_name} — Failure Frames", fontsize=14, y=1.01)

    for i, (before, fail, step) in enumerate(frames_data):
        axes[i, 0].imshow(before)
        axes[i, 0].set_ylabel(f"step {step}", fontsize=10)
        axes[i, 1].imshow(fail)

        for j in range(2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    axes[0, 0].set_title(f"Before (t − {steps_before})", fontsize=11)
    axes[0, 1].set_title("At failure (t)", fontsize=11)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {output_path.with_suffix('.pdf')}")


def main():
    parser = ArgumentParser(description="Explore failure modes visually across environments")
    parser.add_argument('--steps_before', type=int, default=None,
                        help='Number of steps before failure to capture (default: hor from task config)')
    parser.add_argument('--n_failures', type=int, default=5,
                        help='Number of failures to collect per env (default: 5)')
    parser.add_argument('--envs', nargs='+', default=None,
                        help='Environments to explore (default: all fail_pred configs)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed (default: 123)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel workers for upkie (default: -1 = all cores)')
    args = parser.parse_args()

    # Collect fail_pred configs
    fail_pred_keys = [k for k in DATASETS if k.endswith('/fail_pred')]
    if args.envs:
        fail_pred_keys = [k for k in fail_pred_keys
                          if k.split('/')[0] in args.envs]

    if not fail_pred_keys:
        available = [k.split('/')[0] for k in DATASETS if k.endswith('/fail_pred')]
        logger.error(f"No matching envs. Available: {available}")
        return

    output_dir = get_root() / 'results' / 'explore_failures'

    for key in sorted(fail_pred_keys):
        cfg = DATASETS[key]
        env_info = ENV_INFO[cfg.env]

        # Use CLI override or fall back to task config horizon
        steps_before = args.steps_before
        if steps_before is None:
            steps_before = TASK_CONFIGS[cfg.env].hor

        logger.info(f"=== {env_info.display_name} ({key}, steps_before={steps_before}) ===")

        if cfg.platform == 'mujoco':
            frames = run_mujoco_failures(cfg, args.n_failures, steps_before, args.seed)
        elif cfg.platform == 'upkie':
            frames = run_upkie_failures(cfg, args.n_failures, steps_before, args.seed, args.n_jobs)
        else:
            logger.warning(f"Unknown platform {cfg.platform} for {key}, skipping")
            continue

        if not frames:
            logger.warning(f"No failures collected for {key} after {MAX_EPISODES} episodes")
            continue

        logger.info(f"Collected {len(frames)} failures for {cfg.env}")
        compose_figure(frames, env_info.display_name, steps_before,
                       output_dir / cfg.env)

    logger.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    main()
