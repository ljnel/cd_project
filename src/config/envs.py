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
    obs_slice: slice | None = None  # Observable dims (qpos+qvel), None = all


ENV_INFO = {
    'inv_pend': EnvInfo(gym_name="InvertedPendulum-v5", obs_dim=4, act_dim=1, display_name="Inv. Pend.", ctrl_freq=25),
    'hopper': EnvInfo(gym_name="Hopper-v5", obs_dim=11, act_dim=3, display_name="Hopper", ctrl_freq=125),
    'half_cheetah': EnvInfo(gym_name="HalfCheetah-v5", obs_dim=17, act_dim=6, display_name="Half Cheetah", ctrl_freq=20),
    'ant': EnvInfo(gym_name="Ant-v5", obs_dim=105, act_dim=8, display_name="Ant", ctrl_freq=20, obs_slice=slice(0, 27)),
    # Humanoid-v5 observation space (348 dims, default: exclude x,y position):
    #   qpos  [0–21]   (22): z-height, torso quaternion, 17 joint angles
    #   qvel  [22–44]  (23): torso linear vel, torso angular vel, 17 joint vels
    #   cinert[45–174] (130): rigid body mass/inertia (13 bodies × 10), ~constant
    #   cvel  [175–252] (78): CoM velocities (13 bodies × 6)
    #   qfrc_actuator [253–269] (17): actuator constraint forces
    #   cfrc_ext [270–347] (78): external contact forces (13 bodies × 6)
    # Dims 0–44 (qpos+qvel) are the physically meaningful sensor readings.
    'humanoid': EnvInfo(gym_name="Humanoid-v5", obs_dim=348, act_dim=17, display_name="Humanoid", ctrl_freq=67, obs_slice=slice(0, 45)),
    'upkie': EnvInfo(gym_name="Upkie-PyBullet-Pendulum", obs_dim=4, act_dim=1, display_name="Upkie", ctrl_freq=200),
}
