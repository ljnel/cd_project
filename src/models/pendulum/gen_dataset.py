import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

n_eps = 1000
ep_len = 1000
skip = 10  # NB: should divide ep_len

# same as for domain randomization when training RL policy. tuned so that mostly successes result
damp_in_lo, damp_in_hi = 0.8, 1.2
# tuned so that failures occur frequently
damp_out_lo, damp_out_hi = 10, 20
# mix of inlier / outlier data for testing
damp_test_lo, damp_test_hi = 0.8, 20

np.random.seed(0)

class FrictionDR(gym.Wrapper):
    """Domain randomization: scale joint friction (and damping) each episode."""
    def __init__(self, env, damp_lo, damp_hi):
        super().__init__(env)
        m = env.unwrapped.model
        self._base_fric = m.dof_frictionloss.copy()
        self._base_damp = m.dof_damping.copy()
        self.damp_lo, self.damp_hi = damp_lo, damp_hi
    def reset(self, **kwargs):
        m = self.unwrapped.model
        damp = np.random.uniform(self.damp_lo, self.damp_hi)
        m.dof_frictionloss[:] = self._base_fric * damp
        m.dof_damping[:]      = self._base_damp * damp  # helps if frictionloss defaults to 0
        self.damp = damp
        return self.env.reset(**kwargs)


def gen_succ():
    env = FrictionDR(gym.make('InvertedPendulum-v5'), damp_in_lo, damp_in_hi)
    model = SAC.load("sac_inv_pend", env=env)

    rew = np.zeros((n_eps, ep_len))
    trip = np.zeros((n_eps, ep_len, 9))
    param = np.zeros(n_eps)

    for episode in range(n_eps):
        obs, _ = env.reset()
        done = False
        param[episode] = env.damp

        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            next_obs, r, done, truncated, info = env.step(action)

            rew[episode, i] = r
            trip[episode, i] = np.concatenate((obs, action, next_obs))

            obs = next_obs
    
    trip = trip.reshape((-1, 9))[::skip] # (n_ep * ep_len / skip, 9)
    rew = rew.flatten()[::skip]
    env.close()

    return trip, rew, param


def gen_fail():
    "Sample from outlier dist and filter for failures."
    env = FrictionDR(gym.make('InvertedPendulum-v5'), damp_out_lo, damp_out_hi)
    model = SAC.load("sac_inv_pend", env=env)

    rew = np.zeros((n_eps, ep_len))
    trip = np.zeros((n_eps, ep_len, 9))
    param = np.zeros(n_eps)

    episode = 0

    while episode < n_eps:
        obs, _ = env.reset()
        done = False
        param[episode] = env.damp

        for i in range(ep_len):
            action, _states = model.predict(obs, deterministic=True)
            next_obs, r, done, truncated, info = env.step(action)

            rew[episode, i] = r
            trip[episode, i] = np.concatenate((obs, action, next_obs))

            obs = next_obs
        
        if rew[episode, -1] < 1: # this episode was a failure
            episode += 1
    
    trip = trip.reshape((-1, 9))[::skip] # (n_ep * ep_len / skip, 9)
    rew = rew.flatten()[::skip]
    env.close()

    return trip, rew, param

def gen_test():
    env = FrictionDR(gym.make('InvertedPendulum-v5'), damp_test_lo, damp_test_hi)    
    model = SAC.load("sac_inv_pend", env=env)

    rew = np.zeros((n_eps, ep_len))
    trip = np.zeros((n_eps, ep_len, 9))
    param = np.zeros(n_eps)

    for episode in range(n_eps):
        obs, _ = env.reset()
        done = False
        param[episode] = env.damp

        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            next_obs, r, done, truncated, info = env.step(action)

            rew[episode, i] = r
            trip[episode, i] = np.concatenate((obs, action, next_obs))

            obs = next_obs
    
    trip = trip.reshape((-1, 9))[::skip] # (n_ep * ep_len / skip, 9)
    rew = rew.flatten()[::skip]
    env.close()

    return trip, rew, param


def create_train():
    trip, rew, param = gen_succ()
    np.savez(
        'train.npz',
        trip=trip,
        rew=rew,
        param=param
    )
    print('Created success data.')


def create_train2():
    trip, rew, param = gen_test()
    np.savez(
        'train2.npz',
        trip=trip,
        rew=rew,
        param=param
    )
    print('Created train data.')

def create_contr_train():
    trip_succ, r_succ, param_succ = gen_succ()
    trip_fail, r_fail, param_fail = gen_fail()
    np.savez(
        'contr_train.npz',
        trip_succ=trip_succ,
        r_succ=r_succ,
        param_succ=param_succ,
        trip_fail=trip_fail,
        r_fail=r_fail,
        param_fail=param_fail
    )
    print('Created contrastive data.')


def create_test():
    trip, rew, param = gen_test()
    np.savez(
        'test.npz',
        trip=trip,
        rew=rew,
        param=param
    )
    print('Created test data.')

if __name__ == "__main__":
    create_train2()
    create_test()
