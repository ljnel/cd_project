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
    term_cond: str = ""         # Termination condition (LaTeX math)
    term_note: str = ""         # Variable definitions (rendered in \scriptsize)
    term_custom: bool = False   # True if custom (not env default) termination


ENV_INFO = {
    'inv_pend': EnvInfo(
        gym_name="InvertedPendulum-v5", obs_dim=4, act_dim=1,
        display_name="Inv. Pend.", ctrl_freq=25,
        term_cond=r"$\lvert\theta\rvert > 11.5\degree$",
        term_note=r"$\theta$: pole angle",
    ),
    'hopper': EnvInfo(
        gym_name="Hopper-v5", obs_dim=11, act_dim=3,
        display_name="Hopper", ctrl_freq=125,
        term_cond=r"$z < 0.7$ or $\lvert\theta\rvert > 11.5\degree$",
        term_note=r"$z$: height, $\theta$: torso angle",
        term_custom=True,
    ),
    'half_cheetah': EnvInfo(
        gym_name="HalfCheetah-v5", obs_dim=17, act_dim=6,
        display_name="Half Cheetah", ctrl_freq=20,
        term_cond=r"$\lvert\phi\rvert > 51.6\degree$",
        term_note=r"$\phi$: front-tip angle",
        term_custom=True,
    ),
    'ant': EnvInfo(
        gym_name="Ant-v5", obs_dim=105, act_dim=8,
        display_name="Ant", ctrl_freq=20, obs_slice=slice(0, 27),
        term_cond=r"$z \notin [0.2, 1.0]$ or $\alpha > 40\degree$",
        term_note=r"$z$: height, $\alpha$: tilt of body up-axis from vertical",
        term_custom=True,
    ),
    # Humanoid-v5 observation space (348 dims, default: exclude x,y position):
    #   qpos  [0–21]   (22): z-height, torso quaternion, 17 joint angles
    #   qvel  [22–44]  (23): torso linear vel, torso angular vel, 17 joint vels
    #   cinert[45–174] (130): rigid body mass/inertia (13 bodies × 10), ~constant
    #   cvel  [175–252] (78): CoM velocities (13 bodies × 6)
    #   qfrc_actuator [253–269] (17): actuator constraint forces
    #   cfrc_ext [270–347] (78): external contact forces (13 bodies × 6)
    # Dims 0–44 (qpos+qvel) are the physically meaningful sensor readings.
    'humanoid': EnvInfo(
        gym_name="Humanoid-v5", obs_dim=348, act_dim=17,
        display_name="Humanoid", ctrl_freq=67, obs_slice=slice(0, 45),
        term_cond=r"$z \notin [1.0, 2.0]\,\mathrm{m}$",
        term_note=r"$z$: height",
        term_custom=True,
    ),
    'upkie': EnvInfo(
        gym_name="Upkie-PyBullet-Pendulum", obs_dim=4, act_dim=1,
        display_name="Upkie", ctrl_freq=200,
        term_cond=r"$\lvert\theta_\mathrm{p}\rvert > 57.3\degree$",
        term_note=r"$\theta_\mathrm{p}$: pitch angle",
    ),
}
