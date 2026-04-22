import numpy as np


def check_custom_termination(gym_name: str, next_obs) -> bool:
    """Custom failure conditions beyond gymnasium defaults."""
    if gym_name == 'HalfCheetah-v5' and abs(next_obs[1]) > 0.9:
        return True
    if gym_name == 'Ant-v5':
        if not 0.2 <= next_obs[0] <= 1.0:
            return True
        qx, qy = next_obs[2], next_obs[3]
        up_z = 1 - 2 * (qx**2 + qy**2)
        max_tilt_deg = 40 
        return up_z < np.cos(np.radians(max_tilt_deg))
    return False
