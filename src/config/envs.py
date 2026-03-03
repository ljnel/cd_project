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
    display_name: str   # Human-readable name for tables/plots
    ctrl_freq: int      # Control frequency in Hz (1 / dt)


ENV_INFO = {
    'inv_pend': EnvInfo(gym_name="InvertedPendulum-v5", obs_dim=4, act_dim=1, display_name="Inv. Pend.", ctrl_freq=25),
    'hopper': EnvInfo(gym_name="Hopper-v5", obs_dim=11, act_dim=3, display_name="Hopper", ctrl_freq=125),
    'half_cheetah': EnvInfo(gym_name="HalfCheetah-v5", obs_dim=17, act_dim=6, display_name="Half Cheetah", ctrl_freq=20),
    'ant': EnvInfo(gym_name="Ant-v5", obs_dim=105, act_dim=8, display_name="Ant", ctrl_freq=20),
    'humanoid': EnvInfo(gym_name="Humanoid-v5", obs_dim=376, act_dim=17, display_name="Humanoid", ctrl_freq=67),
    'upkie': EnvInfo(gym_name="Upkie-PyBullet-Pendulum", obs_dim=4, act_dim=1, display_name="Upkie", ctrl_freq=200),
}
