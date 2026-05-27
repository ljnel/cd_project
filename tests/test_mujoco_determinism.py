"""Verify gymnasium MuJoCo `env.step` is a bitwise-deterministic function of (state, action).

`np.array_equal` on finite floats is equivalent to bit-equality, which we assert
on entire next-state vectors (max |diff| = 0.0).
"""

import gymnasium as gym
import numpy as np
import pytest

ENV_NAME = 'Hopper-v5'


@pytest.fixture
def env():
    e = gym.make(ENV_NAME, terminate_when_unhealthy=False)
    yield e
    e.close()


def test_same_seed_reset_is_reproducible():
    a = gym.make(ENV_NAME)
    b = gym.make(ENV_NAME)
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    assert np.array_equal(obs_a, obs_b)


def test_clone_state_then_same_action_gives_identical_next_state():
    """The core determinism property: (state, action) -> next_state is a pure function."""
    a = gym.make(ENV_NAME, terminate_when_unhealthy=False)
    b = gym.make(ENV_NAME, terminate_when_unhealthy=False)
    a.reset(seed=0)
    b.reset(seed=999)  # deliberately different starting state

    # Copy a's internal (qpos, qvel) into b.
    b.unwrapped.set_state(a.unwrapped.data.qpos.copy(),
                          a.unwrapped.data.qvel.copy())

    action = np.array([0.1, -0.2, 0.3])
    next_a, *_ = a.step(action)
    next_b, *_ = b.step(action)

    assert np.array_equal(next_a, next_b)
    assert float(np.max(np.abs(next_a - next_b))) == 0.0


def test_replay_same_actions_gives_bitwise_identical_trajectory(env):
    """Replaying the same action sequence from the same reset yields the same trajectory."""
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(500, env.action_space.shape[0]))

    env.reset(seed=42)
    traj1 = np.stack([env.step(a)[0] for a in actions])

    env.reset(seed=42)
    traj2 = np.stack([env.step(a)[0] for a in actions])

    assert np.array_equal(traj1, traj2)
    assert float(np.max(np.abs(traj1 - traj2))) == 0.0


def test_constant_zero_action_long_rollout_is_reproducible(env):
    """Even with the same action every step, the simulator state evolves;
    replaying it must reproduce the trajectory bitwise."""
    n_steps = 2000
    zero = np.zeros(env.action_space.shape[0])

    env.reset(seed=7)
    traj1 = np.stack([env.step(zero)[0] for _ in range(n_steps)])

    env.reset(seed=7)
    traj2 = np.stack([env.step(zero)[0] for _ in range(n_steps)])

    assert np.array_equal(traj1, traj2)
