"""
Disturbance classes for Upkie robot simulation.

Disturbances are per-step perturbations (forces, sensor noise, actuator issues)
that can be applied during episode rollouts to induce failures.

These are mechanisms for generating diverse failure modes, not objects of detection.
"""

import numpy as np
from typing import Tuple, Any, List
from abc import ABC, abstractmethod
from upkie.utils.external_force import ExternalForce


# =============================================================================
# Helper
# =============================================================================

def clear_external_forces(simulator: Any) -> None:
    simulator.set_external_forces({"torso": ExternalForce(np.zeros(3))})


# =============================================================================
# Base Class
# =============================================================================

class Disturbance(ABC):
    """Base class for per-step disturbances."""

    @abstractmethod
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        """Called at the start of each episode."""
        pass

    @abstractmethod
    def apply(self, step: int, obs: np.ndarray, action: np.ndarray,
              simulator: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Apply disturbance at timestep. Returns (obs, action)."""
        pass


# =============================================================================
# Force Disturbances
# =============================================================================

class ConstantForce(Disturbance):
    """Constant external force throughout episode."""

    def __init__(self, force: np.ndarray = None):
        self.force = force if force is not None else np.array([1.0, 0.0, 0.0])
        self.external_force = None

    def reset(self, n_steps, rng):
        self.external_force = ExternalForce(force=self.force)

    def apply(self, step, obs, action, simulator):
        simulator.set_external_forces({"torso": self.external_force})
        return obs, action


class ImpulseForce(Disturbance):
    """Brief impulse force at random time."""

    def __init__(self, force_magnitude: float = 5.0, duration_seconds: float = 0.2,
                 frequency: float = 200.0):
        self.force_magnitude = force_magnitude
        self.duration_steps = int(duration_seconds * frequency)
        self.start_step = 0
        self.end_step = 0
        self.external_force = None

    def reset(self, n_steps, rng):
        latest = max(0, n_steps - self.duration_steps - 10)
        earliest = min(50, latest)
        self.start_step = rng.integers(earliest, max(earliest + 1, latest))
        self.end_step = self.start_step + self.duration_steps
        angle = rng.uniform(0, 2 * np.pi)
        force = self.force_magnitude * np.array([np.cos(angle), np.sin(angle), 0.0])
        self.external_force = ExternalForce(force=force)

    def apply(self, step, obs, action, simulator):
        if self.start_step <= step < self.end_step:
            simulator.set_external_forces({"torso": self.external_force})
        else:
            clear_external_forces(simulator)
        return obs, action


class PeriodicForce(Disturbance):
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


class RandomWalkForce(Disturbance):
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


# =============================================================================
# Sensor Disturbances
# =============================================================================

class SensorNoise(Disturbance):
    """Noise injection into observations."""

    def __init__(self, noise_std: np.ndarray = None):
        self.noise_std = noise_std if noise_std is not None else np.array([0.05, 0.02, 0.1, 0.05])
        self.rng = None

    def reset(self, n_steps, rng):
        self.rng = rng

    def apply(self, step, obs, action, simulator):
        return obs + self.rng.normal(0, self.noise_std), action


class SensorBias(Disturbance):
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


class SensorDropout(Disturbance):
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


# =============================================================================
# Actuator Disturbances
# =============================================================================

class ActuatorDegradation(Disturbance):
    """Scales down actions (motor weakness)."""

    def __init__(self, efficiency: float = 0.5):
        self.efficiency = efficiency

    def reset(self, n_steps, rng):
        pass

    def apply(self, step, obs, action, simulator):
        return obs, action * self.efficiency


class ActuatorNoise(Disturbance):
    """Noise added to actions (motor jitter)."""

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
        self.rng = None

    def reset(self, n_steps, rng):
        self.rng = rng

    def apply(self, step, obs, action, simulator):
        return obs, action + self.rng.normal(0, self.noise_std, size=action.shape)


class ActuatorDelay(Disturbance):
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


class Latency(Disturbance):
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


# =============================================================================
# Composite
# =============================================================================

class CompositeDisturbance(Disturbance):
    """Combines multiple disturbances."""

    def __init__(self, disturbances: List[Disturbance]):
        self.disturbances = disturbances

    def reset(self, n_steps, rng):
        for d in self.disturbances:
            d.reset(n_steps, rng)

    def apply(self, step, obs, action, simulator):
        for d in self.disturbances:
            obs, action = d.apply(step, obs, action, simulator)
        return obs, action
