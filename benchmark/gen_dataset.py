import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import time

n_eps = 1000
ep_len = 1000
skip = 20  # NB: should divide ep_len

"""
Run this file to generate datasets for an environment and policy.
Example:    python gen_dataset.py --env hopper
"""

class DomainRandomizer(gym.Wrapper):
    def __init__(self, env, dof_damping_lo, dof_damping_hi, mass_lo, mass_hi, fric_lo, fric_hi):
        super().__init__(env)
        m = env.unwrapped.model
        # store base values
        self._base_damp = m.dof_damping.copy()  # viscous damping
        self._base_mass = m.body_mass.copy()
        self._base_fric = m.geom_friction.copy()

        self.dof_damping_lo = dof_damping_lo
        self.dof_damping_hi = dof_damping_hi
        self.mass_lo = mass_lo
        self.mass_hi = mass_hi
        self.fric_lo = fric_lo
        self.fric_hi = fric_hi

    def reset(self, **kwargs):
        m = self.unwrapped.model

        # damping randomization
        damp_scale = np.random.uniform(self.dof_damping_lo, self.dof_damping_hi)
        m.dof_damping[:] = self._base_damp * damp_scale

        # mass randomization
        mass_scale = np.random.uniform(self.mass_lo, self.mass_hi)
        m.body_mass[:] = self._base_mass * mass_scale

        # geom friction randomization
        fric_scale = np.random.uniform(self.fric_lo, self.fric_hi)
        m.geom_friction[:] = self._base_fric * fric_scale

        # store for later access
        self.damp = damp_scale
        self.mass = mass_scale
        self.fric = fric_scale

        return self.env.reset(**kwargs)


def gen_data(env, policy, dof_damping_lo, dof_damping_hi, mass_lo, mass_hi, fric_lo, fric_hi):
    cheetah = True if env == 'HalfCheetah-v5' else False
    env = DomainRandomizer(gym.make(env), dof_damping_lo, dof_damping_hi, mass_lo, mass_hi, fric_lo, fric_hi)
    model = SAC.load(policy, env=env)

    s_dim, a_dim = env.observation_space.shape[0], env.action_space.shape[0]

    rew = np.zeros((n_eps, ep_len))
    sa = np.zeros((n_eps, ep_len, s_dim + a_dim))
    param = np.zeros(n_eps)
    fail = np.zeros(n_eps)  # 0 means never failed

    for episode in range(n_eps):
        obs, _ = env.reset()
        terminated = False
        param[episode] = env.damp

        for i in range(1000):
            if terminated and fail[episode] == 0:
                fail[episode] = i

            action, _states = model.predict(obs, deterministic=True)
            next_obs, r, terminated, truncated, info = env.step(action)

            rew[episode, i] = r
            sa[episode, i] = np.concatenate((obs, action))

            # define a failure condition for half cheetah
            if cheetah and abs(obs[1]) > 1.0: # fail if too much tilt
                terminated = True

            obs = next_obs

        if (episode + 1) % 50 == 0:
            print(f"Generated {episode + 1} episodes")
    
    sa = sa.reshape((-1, s_dim + a_dim))[::skip] # (n_ep * ep_len / skip, s_dim + a_dim)
    rew = rew.flatten()[::skip]
    fail /= skip
    env.close()

    return sa, rew, param, fail


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument('--env')
    args = parser.parse_args()

    if args.env == 'inv_pend':
        env = 'InvertedPendulum-v5'
        model = 'invertedpendulum-v5-sac-expert.zip'
        #model = 'sac_inv_pend'
        dof_damping_lo, dof_damping_hi = 0.6, 20.
        mass_lo, mass_hi = .6, 2.4
        fric_lo, fric_hi = .6, 2.4

    elif args.env == 'hopper':
        env = 'Hopper-v5'
        #model = 'sac_hopper'
        model = 'hopper-v5-sac-expert.zip'
        dof_damping_lo, dof_damping_hi = 0.9, 1.1
        mass_lo, mass_hi = .9, 1.1
        fric_lo, fric_hi = .9, 1.1

    elif args.env == 'half_cheetah':
        env = 'HalfCheetah-v5'
        model = 'halfcheetah-v5-sac-expert.zip'
        dof_damping_lo, dof_damping_hi = 0.4, 2.8
        mass_lo, mass_hi = .4, 2.
        fric_lo, fric_hi = .4, 2.

    elif args.env == 'ant':
        env = 'Ant-v5'
        model = 'ant-v5-sac-expert.zip'
        dof_damping_lo, dof_damping_hi = 0.6, 2.8
        mass_lo, mass_hi = .6, 1.8
        fric_lo, fric_hi = .6, 1.4

    elif args.env == 'humanoid':
        env = 'Humanoid-v5'
        model = 'humanoid-v5-sac-expert.zip'
        dof_damping_lo, dof_damping_hi = 0.8, 1.2
        mass_lo, mass_hi = .8, 1.2
        fric_lo, fric_hi = .8, 1.2
    dir = Path(f'./{args.env}')

    np.random.seed(0)
    start = time.time()
    sa, rew, param, fail = gen_data(env, dir / model, dof_damping_lo, dof_damping_hi, mass_lo, mass_hi, fric_lo, fric_hi)
    end = time.time()
    np.savez(dir / 'train.npz',
             sa=sa,
             rew=rew,
             param=param,
             fail=fail
    )
    print(f'Created train data with {np.mean(fail > 0)} fails in {end - start} seconds.')

    np.random.seed(1)
    start = time.time()
    sa, rew, param, fail = gen_data(env, dir / model, dof_damping_lo, dof_damping_hi, mass_lo, mass_hi, fric_lo, fric_hi)
    end = time.time()
    np.savez(dir / 'test.npz',
             sa=sa,
             rew=rew,
             param=param,
             fail=fail
    )
    print(f'Created test data with {np.mean(fail > 0)} fails in {end - start} seconds.')

    