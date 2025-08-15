from .pendulum_model import create
import gymnasium as gym
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


class PendulumEnv(gym.Env):
    """
    Create a gym wrapper for a pendulum simulated in pinocchio.
    Right now, this samples damping in [b_lo, b_hi], but eventually it should sample more params.

    NB: right now 200 steps are used
    """

    def __init__(self, time, freq, b_lo=0., b_hi=.1, n_pend=1):
        super().__init__()
        self.model, self.geom_model = create(n_pend)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nsteps = 0
        self.b_lo, self.b_hi = b_lo, b_hi

        obs_lo = np.concatenate([-np.ones(2*self.nq), -np.inf * np.ones(self.nq)])
        self.observation_space = gym.spaces.Box(low=obs_lo,
                                                high=-obs_lo,
                                                shape=(2 * self.nq + self.nv,),
                                                dtype=np.float64)
        action_lo = -2 * np.ones(self.nv)
        self.action_space = gym.spaces.Box(low=action_lo, high=-action_lo, 
                                           shape=(self.nv,), dtype=np.float64)
        self.dt = 1 / freq

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.nsteps = 0
        self.q = 2 * np.pi * np.random.rand(self.nq) - np.pi  # [-pi, pi]
        self.v = 2 * np.random.rand(self.nv) - 1  # [-1, 1]
        obs = np.concatenate([np.cos(self.q), np.sin(self.q), self.v])

        self.b = np.random.rand() * (self.b_hi - self.b_lo) + self.b_lo
        return obs, {}
    
    def step(self, action):
        tau = action - self.b * self.v
        a = pin.aba(self.model, self.data, self.q, self.v, tau)
        self.v += a * self.dt
        self.q = pin.integrate(self.model, self.q, self.v * self.dt)

        obs = np.concatenate([np.cos(self.q), np.sin(self.q), self.v])
        norm_q = (self.q + np.pi) % (2 * np.pi) - np.pi  # NB
        reward = -np.sum(norm_q ** 2 + 0.1 * self.v ** 2 + 0.001 * tau ** 2)
        self.nsteps += 1
        truncated = self.nsteps >= 200

        return obs, reward, False, truncated, {}
    
    def get_viz(self):
        viz = MeshcatVisualizer(self.model, self.geom_model, self.geom_model)
        viz.initViewer(loadModel=True)
        return viz