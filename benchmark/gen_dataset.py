import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import time
from argparse import ArgumentParser
from pathlib import Path

n_eps = 1000
ep_len = 1000

"""
Run this file to generate datasets for an environment and policy.
Example:    python gen_dataset.py --env hopper
"""

class DomainRandomizer(gym.Wrapper):
    def __init__(self, env, dof_damping_lo, dof_damping_hi, mass_lo, mass_hi, fric_lo, fric_hi):
        super().__init__(env)
        self.m = env.unwrapped.model
        # store base values
        self._base_damp = self.m.dof_damping.copy()  # viscous damping
        self._base_mass = self.m.body_mass.copy()
        self._base_fric = self.m.geom_friction.copy()

        self.dof_damping_lo = dof_damping_lo
        self.dof_damping_hi = dof_damping_hi
        self.mass_lo = mass_lo
        self.mass_hi = mass_hi
        self.fric_lo = fric_lo
        self.fric_hi = fric_hi

    def reset(self, **kwargs):
        damp_scale = np.random.uniform(self.dof_damping_lo, self.dof_damping_hi)
        mass_scale = np.random.uniform(self.mass_lo, self.mass_hi)
        fric_scale = np.random.uniform(self.fric_lo, self.fric_hi)

        self.m.dof_damping[:] = self._base_damp * damp_scale
        self.m.body_mass[:] = self._base_mass * mass_scale
        self.m.geom_friction[:] = self._base_fric * fric_scale

        # store for later access
        self.damp = damp_scale
        self.mass = mass_scale
        self.fric = fric_scale

        return self.env.reset(**kwargs)


def gen_data(env, policy, rng, dof_damping_lo, dof_damping_hi, mass_lo, mass_hi, fric_lo, fric_hi):

    cheetah = True if env == 'HalfCheetah-v5' else False
    env = gym.make(env)
    model = SAC.load(policy, env=env)
    s_dim, a_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # generate randomness
    seeds = rng.integers(0, 2**32, size=n_eps, dtype=np.uint32)
    damp = rng.uniform(dof_damping_lo, dof_damping_hi, size=n_eps)
    mass = rng.uniform(mass_lo, mass_hi, size=n_eps)
    fric = rng.uniform(fric_lo, fric_hi, size=n_eps)

    # set up some other arrays
    states = np.zeros((n_eps, ep_len, s_dim))
    actions = np.zeros((n_eps, ep_len, a_dim))
    rew = np.zeros((n_eps, ep_len))
    fail = np.zeros((n_eps, ep_len))

    m = env.unwrapped.model
    base_damp = m.dof_damping.copy()  # viscous damping
    base_mass = m.body_mass.copy()
    base_fric = m.geom_friction.copy()

    for episode in range(n_eps):

        m.dof_damping[:] = base_damp * damp[episode]
        m.body_mass[:] = base_mass * mass[episode]
        m.geom_friction[:] = base_fric * fric[episode]
        obs, _ = env.reset(seed=int(seeds[episode]))
        terminated = False

        for i in range(ep_len):
            fail[episode, i] = terminated

            action, _states = model.predict(obs, deterministic=True)
            next_obs, r, terminated, truncated, info = env.step(action)

            # define a failure condition for half cheetah
            if cheetah and abs(next_obs[1]) > .9: # fail if too much tilt
                terminated = True

            rew[episode, i] = r
            states[episode, i] = obs
            actions[episode, i] = action
            obs = next_obs

        if (episode + 1) % 50 == 0:
            print(f"Generated {episode + 1} episodes")
    
    #sa = sa.reshape((-1, s_dim + a_dim))[::skip] # (n_ep * ep_len / skip, s_dim + a_dim)
    #rew = rew.flatten()[::skip]
    #fail /= skip
    env.close()

    fail = np.where(fail.argmax(axis=1) == 0, 0, fail.argmax(axis=1))  # first failure step or ep_len
    params = np.stack([damp, mass, fric], axis=1)
    return states, actions, rew, fail, params, seeds


if __name__ == "__main__":    

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
        dof_damping_lo, dof_damping_hi = 1., 1.
        mass_lo, mass_hi = 1., 1.
        fric_lo, fric_hi = 1., 1.

    elif args.env == 'half_cheetah':
        env = 'HalfCheetah-v5'
        model = 'halfcheetah-v5-sac-expert.zip'
        dof_damping_lo, dof_damping_hi = 0.4, 2.8
        mass_lo, mass_hi = .4, 2.2
        fric_lo, fric_hi = .4, 2.2

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

    rng_train = np.random.default_rng(0)
    start = time.time()
    states, actions, rew, fail, params, seeds = gen_data(env, dir / model, rng_train,
                                                  dof_damping_lo, 
                                                  dof_damping_hi, 
                                                  mass_lo, 
                                                  mass_hi, 
                                                  fric_lo, 
                                                  fric_hi)
    end = time.time()
    np.savez(dir / 'train.npz',
            states=states,
            actions=actions,
            rew=rew,
            fail=fail,
            params=params,
            seeds=seeds
    )
    print(f'Created train data with {np.mean(fail > 0)} fails in {end - start} seconds.')

    rng_test = np.random.default_rng(1)
    start = time.time()
    states, actions, rew, fail, params, seeds = gen_data(env, dir / model, rng_test,
                                                         dof_damping_lo, 
                                                         dof_damping_hi, 
                                                         mass_lo, 
                                                         mass_hi, 
                                                         fric_lo, 
                                                         fric_hi)
    end = time.time()
    np.savez(dir / 'test.npz',
            states=states,
            actions=actions,
            rew=rew,
            fail=fail,
            params=params,
            seeds=seeds
    )
    print(f'Created test data with {np.mean(fail > 0)} fails in {end - start} seconds.')