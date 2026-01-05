import numpy as np
import gymnasium as gym
from config.envs import EnvConfig


class EnvRandomizer(gym.Wrapper):
    def __init__(self, env: gym.Env, cfg: EnvConfig):
        super().__init__(env)
        self.m = env.unwrapped.model
        # store base values
        self._base_damp = self.m.dof_damping.copy()  # viscous damping
        self._base_mass = self.m.body_mass.copy()
        self._base_fric = self.m.geom_friction.copy()

        self.dof_damping_lo, self.dof_damping_hi = cfg.dof_damping
        self.mass_lo, self.mass_hi = cfg.mass
        self.fric_lo, self.fric_hi = cfg.fric

    def reset(self, **kwargs):
        damp_scale = np.random.uniform(
            self.dof_damping_lo, self.dof_damping_hi)
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
