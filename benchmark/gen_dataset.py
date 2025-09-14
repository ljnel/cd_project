import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

n_eps = 1000
ep_len = 1000
skip = 20  # NB: should divide ep_len

np.random.seed(0)

"""
Run this file to generate datasets for an environment and policy.
Example:    python gen_dataset.py --env hopper
"""

dof_damping_lo, dof_damping_hi = 0.6, 1.4
body_mass_lo, body_mass_hi = 1., 1.  # change these later
geom_friction_lo, geom_friction_hi = 1., 1.


class DomainRandomizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        m = env.unwrapped.model
        # store base values
        self._base_fric = m.dof_frictionloss.copy()
        self._base_damp = m.dof_damping.copy()

    def reset(self, **kwargs):
        m = self.unwrapped.model
        damp = np.random.uniform(dof_damping_lo, dof_damping_hi)
        #mass = 
        #fric = 

        m.dof_frictionloss[:] = self._base_fric * damp
        m.dof_damping[:]      = self._base_damp * damp  # helps if frictionloss defaults to 0
        # so that params can be accessed during rollouts
        self.damp = damp
        #self.mass = mass
        #self.fric = fric
        return self.env.reset(**kwargs)


def gen_data(env, policy):
    env = DomainRandomizer(gym.make(env))
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
    parser.add_argument('--env', type=str, default='hopper')
    args = parser.parse_args()

    if args.env == 'inv_pend':
        env = 'InvertedPendulum-v5'
        model = 'sac_inv_pend'
    if args.env == 'hopper':
        env = 'Hopper-v5'
        model = 'sac_hopper'
    if args.env == 'half_cheetah':
        env = 'HalfCheetah-v5'
        model = 'halfcheetah-v5-sac-medium'
    if args.env == 'ant':
        env = 'Ant-v5'
        model = 'ant-v5-sac-expert.zip'
    if args.env == 'humanoid':
        raise NotImplementedError
    dir = Path(f'./{args.env}')

    sa, rew, param, fail = gen_data(env, dir / model)
    np.savez(dir / 'train.npz',
             sa=sa,
             rew=rew,
             param=param,
             fail=fail
    )
    print(f'Created train data with {np.mean(fail > 0)} fails')

    sa, rew, param, fail = gen_data(env, dir / model)
    np.savez(dir / 'test.npz',
             sa=sa,
             rew=rew,
             param=param,
             fail=fail
    )
    print(f'Created test data with {np.mean(fail > 0)} fails.')

    