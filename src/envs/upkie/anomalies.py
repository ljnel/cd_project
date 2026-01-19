"""
Anomaly classes for Upkie robot simulation.

Two types:
- Per-step anomalies (Anomaly): Forces, sensor noise, actuator issues
- Parameter anomalies (ParameterAnomaly): Mass, friction, damping changes

Per-step anomalies implement is_active(step) to indicate when they're active.
Parameter anomalies sample from ranges and report sampled values via get_sampled_scales().
"""

import numpy as np
import pybullet as p
from typing import Tuple, Any, List
from abc import ABC, abstractmethod
from upkie.utils.external_force import ExternalForce


# =============================================================================
# Helper
# =============================================================================

def clear_external_forces(simulator: Any) -> None:
    simulator.set_external_forces({"torso": ExternalForce(np.zeros(3))})


# =============================================================================
# Per-Step Anomalies
# =============================================================================

class Anomaly(ABC):
    """Base class for per-step anomalies."""

    @abstractmethod
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        """Called at the start of each anomaly episode."""
        pass

    @abstractmethod
    def apply(self, step: int, obs: np.ndarray, action: np.ndarray,
              simulator: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Apply anomaly at timestep. Returns (obs, action)."""
        pass

    def is_active(self, step: int) -> bool:
        """Whether anomaly is active at this step. Default: always active."""
        return True


class ConstantForceAnomaly(Anomaly):
    """Constant external force throughout episode."""

    def __init__(self, force: np.ndarray = None):
        self.force = force if force is not None else np.array([1.0, 0.0, 0.0])
        self.external_force = None

    def reset(self, n_steps, rng):
        self.external_force = ExternalForce(force=self.force)

    def apply(self, step, obs, action, simulator):
        simulator.set_external_forces({"torso": self.external_force})
        return obs, action


class ImpulseForceAnomaly(Anomaly):
    """Brief impulse force at random time."""

    def __init__(self, force_magnitude: float = 5.0, duration_seconds: float = 0.2,
                 frequency: float = 200.0):
        self.force_magnitude = force_magnitude
        self.duration_steps = int(duration_seconds * frequency)
        self.start_step = 0
        self.external_force = None

    def reset(self, n_steps, rng):
        latest = max(0, n_steps - self.duration_steps - 10)
        earliest = min(50, latest)
        self.start_step = rng.integers(earliest, max(earliest + 1, latest))
        angle = rng.uniform(0, 2 * np.pi)
        force = self.force_magnitude * np.array([np.cos(angle), np.sin(angle), 0.0])
        self.external_force = ExternalForce(force=force)

    def apply(self, step, obs, action, simulator):
        if self.is_active(step):
            simulator.set_external_forces({"torso": self.external_force})
        else:
            clear_external_forces(simulator)
        return obs, action

    def is_active(self, step):
        return self.start_step <= step < self.start_step + self.duration_steps


class PeriodicForceAnomaly(Anomaly):
    """Oscillating force (vibration/wobble)."""

    def __init__(self, amplitude: float = 2.0, frequency_hz: float = 2.0,
                 control_frequency: float = 200.0):
        self.amplitude = amplitude
        self.omega = 2 * np.pi * frequency_hz
        self.dt = 1.0 / control_frequency
        self.phase = 0.0
        self.direction = np.array([1.0, 0.0, 0.0])

    def reset(self, n_steps, rng):
        self.phase = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, 2 * np.pi)
        self.direction = np.array([np.cos(angle), np.sin(angle), 0.0])

    def apply(self, step, obs, action, simulator):
        t = step * self.dt
        force = self.amplitude * np.sin(self.omega * t + self.phase) * self.direction
        simulator.set_external_forces({"torso": ExternalForce(force=force)})
        return obs, action


class RandomWalkForceAnomaly(Anomaly):
    """Smoothly varying random force (wind gusts)."""

    def __init__(self, max_magnitude: float = 3.0, smoothness: float = 0.95):
        self.max_magnitude = max_magnitude
        self.smoothness = smoothness
        self.current_force = np.zeros(3)
        self.rng = None

    def reset(self, n_steps, rng):
        self.rng = rng
        self.current_force = np.zeros(3)

    def apply(self, step, obs, action, simulator):
        noise = self.rng.normal(0, 1, size=3)
        noise[2] = 0
        self.current_force = self.smoothness * self.current_force + (1 - self.smoothness) * noise
        mag = np.linalg.norm(self.current_force)
        if mag > 1.0:
            self.current_force /= mag
        force = self.current_force * self.max_magnitude
        simulator.set_external_forces({"torso": ExternalForce(force=force)})
        return obs, action


# --- Sensor Anomalies ---

class SensorNoiseAnomaly(Anomaly):
    """Noise injection into observations."""

    def __init__(self, noise_std: np.ndarray = None):
        self.noise_std = noise_std if noise_std is not None else np.array([0.05, 0.02, 0.1, 0.05])
        self.rng = None

    def reset(self, n_steps, rng):
        self.rng = rng

    def apply(self, step, obs, action, simulator):
        return obs + self.rng.normal(0, self.noise_std), action


class SensorBiasAnomaly(Anomaly):
    """Constant bias added to observations (sensor drift)."""

    def __init__(self, bias: np.ndarray = None, random_scale: np.ndarray = None):
        self.fixed_bias = bias
        self.random_scale = random_scale if random_scale is not None else np.array([0.1, 0.05, 0.0, 0.0])
        self.bias = None

    def reset(self, n_steps, rng):
        self.bias = self.fixed_bias if self.fixed_bias is not None else \
                    rng.uniform(-1, 1, size=len(self.random_scale)) * self.random_scale

    def apply(self, step, obs, action, simulator):
        return obs + self.bias, action


class SensorDropoutAnomaly(Anomaly):
    """Random sensor dropouts."""

    def __init__(self, dropout_prob: float = 0.1, use_last: bool = True):
        self.dropout_prob = dropout_prob
        self.use_last = use_last
        self.last_obs = None
        self.rng = None

    def reset(self, n_steps, rng):
        self.rng = rng
        self.last_obs = None

    def apply(self, step, obs, action, simulator):
        if self.rng.random() < self.dropout_prob:
            return (self.last_obs.copy() if self.use_last and self.last_obs is not None
                    else np.zeros_like(obs)), action
        self.last_obs = obs.copy()
        return obs, action


# --- Actuator Anomalies ---

class ActuatorDegradationAnomaly(Anomaly):
    """Scales down actions (motor weakness)."""

    def __init__(self, efficiency: float = 0.5):
        self.efficiency = efficiency

    def reset(self, n_steps, rng):
        pass

    def apply(self, step, obs, action, simulator):
        return obs, action * self.efficiency


class ActuatorNoiseAnomaly(Anomaly):
    """Noise added to actions (motor jitter)."""

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
        self.rng = None

    def reset(self, n_steps, rng):
        self.rng = rng

    def apply(self, step, obs, action, simulator):
        return obs, action + self.rng.normal(0, self.noise_std, size=action.shape)


class ActuatorDelayAnomaly(Anomaly):
    """Delayed action application."""

    def __init__(self, delay_steps: int = 3):
        self.delay_steps = delay_steps
        self.buffer = []

    def reset(self, n_steps, rng):
        self.buffer = []

    def apply(self, step, obs, action, simulator):
        self.buffer.append(action.copy())
        if len(self.buffer) > self.delay_steps:
            return obs, self.buffer.pop(0)
        return obs, np.zeros_like(action)


class LatencyAnomaly(Anomaly):
    """Delayed observations."""

    def __init__(self, delay_steps: int = 5):
        self.delay_steps = delay_steps
        self.buffer = []

    def reset(self, n_steps, rng):
        self.buffer = []

    def apply(self, step, obs, action, simulator):
        self.buffer.append(obs.copy())
        if len(self.buffer) > self.delay_steps:
            return self.buffer.pop(0), action
        return self.buffer[0], action


class CompositeAnomaly(Anomaly):
    """Combines multiple per-step anomalies."""

    def __init__(self, anomalies: List[Anomaly]):
        self.anomalies = anomalies

    def reset(self, n_steps, rng):
        for a in self.anomalies:
            a.reset(n_steps, rng)

    def apply(self, step, obs, action, simulator):
        for a in self.anomalies:
            obs, action = a.apply(step, obs, action, simulator)
        return obs, action

    def is_active(self, step):
        return any(a.is_active(step) for a in self.anomalies)


# =============================================================================
# Parameter Anomalies (PyBullet-specific)
# =============================================================================

class ParameterAnomaly(ABC):
    """Base class for parameter anomalies. Samples from ranges at episode start."""

    @abstractmethod
    def apply_at_reset(self, robot_id: int, rng: np.random.Generator) -> None:
        pass

    @abstractmethod
    def restore(self, robot_id: int) -> None:
        pass

    @abstractmethod
    def get_sampled_scales(self) -> dict:
        """Returns dict of param_name -> sampled value for this episode."""
        pass


class MassAnomaly(ParameterAnomaly):
    """Scales mass of robot links. Samples from range each episode."""

    def __init__(self, mass_range: Tuple[float, float] = (1.2, 1.8), link_indices: list = None):
        self.mass_range = mass_range
        self.link_indices = link_indices
        self.base_masses = {}
        self.sampled_scale = 1.0

    def apply_at_reset(self, robot_id, rng):
        self.sampled_scale = rng.uniform(*self.mass_range)
        n = p.getNumJoints(robot_id)
        indices = self.link_indices if self.link_indices else list(range(-1, n))
        for idx in indices:
            mass = p.getDynamicsInfo(robot_id, idx)[0]
            self.base_masses[idx] = mass
            p.changeDynamics(robot_id, idx, mass=mass * self.sampled_scale)

    def restore(self, robot_id):
        for idx, mass in self.base_masses.items():
            p.changeDynamics(robot_id, idx, mass=mass)
        self.base_masses = {}

    def get_sampled_scales(self):
        return {'mass_scale': self.sampled_scale}


class FrictionAnomaly(ParameterAnomaly):
    """Modifies friction coefficients. Samples from range each episode."""

    def __init__(self, friction_range: Tuple[float, float] = (0.2, 0.6), link_indices: list = None):
        self.friction_range = friction_range
        self.link_indices = link_indices
        self.base_frictions = {}
        self.sampled_scale = 1.0

    def apply_at_reset(self, robot_id, rng):
        self.sampled_scale = rng.uniform(*self.friction_range)
        n = p.getNumJoints(robot_id)
        indices = self.link_indices if self.link_indices else list(range(-1, n))
        for idx in indices:
            fric = p.getDynamicsInfo(robot_id, idx)[1]
            self.base_frictions[idx] = fric
            p.changeDynamics(robot_id, idx, lateralFriction=fric * self.sampled_scale)

    def restore(self, robot_id):
        for idx, fric in self.base_frictions.items():
            p.changeDynamics(robot_id, idx, lateralFriction=fric)
        self.base_frictions = {}

    def get_sampled_scales(self):
        return {'friction_scale': self.sampled_scale}


class JointDampingAnomaly(ParameterAnomaly):
    """Modifies joint damping. Samples from range each episode."""

    def __init__(self, damping_range: Tuple[float, float] = (2.0, 4.0), joint_indices: list = None):
        self.damping_range = damping_range
        self.joint_indices = joint_indices
        self.base_dampings = {}
        self.sampled_scale = 1.0

    def apply_at_reset(self, robot_id, rng):
        self.sampled_scale = rng.uniform(*self.damping_range)
        n = p.getNumJoints(robot_id)
        indices = self.joint_indices if self.joint_indices else list(range(n))
        for idx in indices:
            damp = p.getJointInfo(robot_id, idx)[6]
            self.base_dampings[idx] = damp
            p.changeDynamics(robot_id, idx, jointDamping=damp * self.sampled_scale)

    def restore(self, robot_id):
        for idx, damp in self.base_dampings.items():
            p.changeDynamics(robot_id, idx, jointDamping=damp)
        self.base_dampings = {}

    def get_sampled_scales(self):
        return {'damping_scale': self.sampled_scale}


class CompositeParameterAnomaly(ParameterAnomaly):
    """Combines multiple parameter anomalies."""

    def __init__(self, anomalies: List[ParameterAnomaly]):
        self.anomalies = anomalies

    def apply_at_reset(self, robot_id, rng):
        for a in self.anomalies:
            a.apply_at_reset(robot_id, rng)

    def restore(self, robot_id):
        for a in self.anomalies:
            a.restore(robot_id)

    def get_sampled_scales(self):
        result = {}
        for a in self.anomalies:
            result.update(a.get_sampled_scales())
        return result