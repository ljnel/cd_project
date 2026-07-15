import numpy as np


def check_custom_termination(gym_name: str, next_obs) -> bool:
    """Failure conditions evaluated from the observation alone.

    For HalfCheetah and Ant these are custom conditions beyond the gym
    defaults. For Hopper and Humanoid they *reproduce* the gym default
    health checks: generation disables the env's built-in termination
    (``terminate_when_unhealthy=False``) so it can keep stepping past failure
    and record post-failure observations, which means the failure index must
    be derived here instead. Indices below assume the default v5 observation
    layout (current x/y position excluded).
    """
    if gym_name == 'HalfCheetah-v5' and abs(next_obs[1]) > 0.9:
        return True
    if gym_name == 'Hopper-v5':
        # gym default: healthy iff 0.7 < z and |angle| < 0.2 rad (≈11.5°).
        z, angle = next_obs[0], next_obs[1]
        return z <= 0.7 or abs(angle) >= 0.2
    if gym_name == 'Humanoid-v5':
        # gym default: healthy iff 1.0 < z < 2.0.
        z = next_obs[0]
        return not (1.0 < z < 2.0)
    if gym_name == 'Ant-v5':
        if not 0.2 <= next_obs[0] <= 1.0:
            return True
        qx, qy = next_obs[2], next_obs[3]
        up_z = 1 - 2 * (qx**2 + qy**2)
        max_tilt_deg = 45
        return up_z < np.cos(np.radians(max_tilt_deg))
    return False
