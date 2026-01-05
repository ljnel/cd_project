import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import time
from argparse import ArgumentParser
from pathlib import Path
from config.envs import ENV_CFG
from utils.paths import get_root

N_EPS = 1000
EP_LEN = 1000

"""
Run this file to generate datasets for an environment and policy.
Example:    python gen_dataset.py --env hopper
"""


def gen_data(cfg, rng):

    env = gym.make(cfg.name)
    policy_path = get_root() / 'src/policies' / cfg.policy
    policy = SAC.load(policy_path, env=env)
    cheetah = True if cfg.name == 'HalfCheetah-v5' else False
    s_dim, a_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # generate randomness
    seeds = rng.integers(0, 2**32, size=N_EPS, dtype=np.uint32)
    damp = rng.uniform(cfg.dof_damping[0], cfg.dof_damping[1], size=N_EPS)
    mass = rng.uniform(cfg.mass[0], cfg.mass[1], size=N_EPS)
    fric = rng.uniform(cfg.fric[0], cfg.fric[1], size=N_EPS)

    # set up some other arrays
    states = np.zeros((N_EPS, EP_LEN, s_dim))
    actions = np.zeros((N_EPS, EP_LEN, a_dim))
    rew = np.zeros((N_EPS, EP_LEN))
    fail = np.zeros((N_EPS, EP_LEN))

    m = env.unwrapped.model
    base_damp = m.dof_damping.copy()  # viscous damping
    base_mass = m.body_mass.copy()
    base_fric = m.geom_friction.copy()

    for episode in range(N_EPS):

        m.dof_damping[:] = base_damp * damp[episode]
        m.body_mass[:] = base_mass * mass[episode]
        m.geom_friction[:] = base_fric * fric[episode]
        obs, _ = env.reset(seed=int(seeds[episode]))
        terminated = False

        for i in range(EP_LEN):
            fail[episode, i] = terminated

            action, _states = policy.predict(obs, deterministic=True)
            next_obs, r, terminated, truncated, info = env.step(action)

            # define a failure condition for half cheetah
            if cheetah and abs(next_obs[1]) > .9:  # fail if too much tilt
                terminated = True

            rew[episode, i] = r
            states[episode, i] = obs
            actions[episode, i] = action
            obs = next_obs

        if (episode + 1) % 50 == 0:
            print(f"Generated {episode + 1} episodes")

    env.close()

    fail = np.where(fail.argmax(axis=1) == 0, -1, fail.argmax(
        axis=1))  # ???
    params = np.stack([damp, mass, fric], axis=1)
    return states, actions, rew, fail, params, seeds


if __name__ == "__main__":
    """
    fetch policy from src/policies
    store data in data/
    """

    parser = ArgumentParser()
    parser.add_argument('--env')
    args = parser.parse_args()

    assert args.env in ENV_CFG.keys()

    rng_train = np.random.default_rng(0)
    start = time.time()
    states, actions, rew, fail, params, seeds = gen_data(ENV_CFG[args.env], rng_train)
    end = time.time()
    np.savez(get_root() / 'data' / args.env / 'train.npz',
             states=states,
             actions=actions,
             rew=rew,
             fail=fail,
             params=params,
             seeds=seeds
             )
    print(
        f'Created train data with {np.mean(fail > 0)} fails in {end - start} seconds.')
    
    rng_test = np.random.default_rng(1)
    start = time.time()
    states, actions, rew, fail, params, seeds = gen_data(ENV_CFG[args.env], rng_test)
    end = time.time()
    np.savez(get_root() / 'data' / args.env / 'test.npz',
             states=states,
             actions=actions,
             rew=rew,
             fail=fail,
             params=params,
             seeds=seeds
             )
    print(
        f'Created test data with {np.mean(fail > 0)} fails in {end - start} seconds.')