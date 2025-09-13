import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

n_eps = 1000
ep_len = 1000
skip = 20  # NB: should divide ep_len

np.random.seed(0)

dof_damping_lo, dof_damping_hi = 0.6, 1.4
body_mass_lo, body_mass_hi = 1., 1.  # change these later
geom_friction_lo, geom_friction_hi = 1., 1.


class RobotDR(gym.Wrapper):
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


def gen_data():
    env = RobotDR(gym.make('Hopper-v5'))
    model = SAC.load("sac_hopper", env=env)

    rew = np.zeros((n_eps, ep_len))
    sa = np.zeros((n_eps, ep_len, 14))
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
    
    sa = sa.reshape((-1, 14))[::skip] # (n_ep * ep_len / skip, 14)
    rew = rew.flatten()[::skip]
    fail /= skip
    env.close()

    return sa, rew, param, fail


def gen_succ_data():
    pass


def create_train():
    sa, rew, param, fail = gen_data()
    np.savez(
        'train.npz',
        sa=sa,
        rew=rew,
        param=param,
        fail=fail
    )
    print(f'Created train data with {np.mean(fail > 0)} fails.')


def create_test():
    sa, rew, param, fail = gen_data()
    np.savez(
        'test.npz',
        sa=sa,
        rew=rew,
        param=param,
        fail=fail
    )
    print(f'Created test data with {np.mean(fail > 0)} fails.')


if __name__ == "__main__":
    create_train()
    create_test()