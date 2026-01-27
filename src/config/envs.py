from dataclasses import dataclass


# =============================================================================
# Environment Info (static, platform-independent)
# =============================================================================

@dataclass
class EnvInfo:
    """Static environment information."""
    gym_name: str       # Gymnasium environment name (e.g., "Hopper-v5")
    obs_dim: int        # Observation dimension
    act_dim: int        # Action dimension


ENV_INFO = {
    'inv_pend': EnvInfo(gym_name="InvertedPendulum-v5", obs_dim=4, act_dim=1),
    'hopper': EnvInfo(gym_name="Hopper-v5", obs_dim=11, act_dim=3),
    'half_cheetah': EnvInfo(gym_name="HalfCheetah-v5", obs_dim=17, act_dim=6),
    'ant': EnvInfo(gym_name="Ant-v5", obs_dim=27, act_dim=8),
    'humanoid': EnvInfo(gym_name="Humanoid-v5", obs_dim=376, act_dim=17),
    'upkie': EnvInfo(gym_name="Upkie-PyBullet-Pendulum", obs_dim=4, act_dim=1),
}
