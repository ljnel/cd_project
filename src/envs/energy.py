"""Mechanical energy of MuJoCo states reconstructed from observations.

The observation stores the generalized coordinates but not the model's inertial
parameters, so energy is recovered by replaying each state through the MuJoCo
model: set (qpos, qvel), run `mj_forward`, and read MuJoCo's energy accounting.
Conservative rigid-body energy only — gravitational PE + 0.5 vᵀM(q)v; actuator
work and contact dissipation are not included.
"""

import gymnasium as gym
import mujoco
import numpy as np


def _ant_model():
    """Ant-v5 mjModel with energy accounting enabled.

    Valid for datasets generated at unit mass scaling (the model's default
    masses); friction/damping vary the *dissipative* dynamics but not the
    conservative energy, so they need no adjustment here.
    """
    env = gym.make('Ant-v5')
    model = env.unwrapped.model
    env.close()
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    return model


def ant_energy(obs: np.ndarray) -> np.ndarray:
    """Rigid-body (potential, kinetic) energy for Ant observations.

    `obs`: (..., D) in the Ant-v5 `obs_slice` layout — `obs[..., 0:13]` is qpos
    with the x,y root position dropped (z, body quaternion, 8 joint angles) and
    `obs[..., 13:27]` is the full qvel. Energy is invariant to x,y, which are
    set to 0 on reconstruction.

    Returns `(..., 2)`: `[potential, kinetic]` in joules. Any non-finite state
    (e.g. a NaN-padded slot) maps to NaN.
    """
    obs = np.asarray(obs)
    model = _ant_model()
    data = mujoco.MjData(model)
    flat = obs.reshape(-1, obs.shape[-1])
    out = np.full((flat.shape[0], 2), np.nan)
    for i, o in enumerate(flat):
        if not np.isfinite(o[:27]).all():
            continue
        data.qpos[:] = np.concatenate([[0.0, 0.0], o[0:13]])
        data.qvel[:] = o[13:27]
        mujoco.mj_forward(model, data)
        out[i] = data.energy
    return out.reshape(*obs.shape[:-1], 2)
