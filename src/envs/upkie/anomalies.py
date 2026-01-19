"""
Anomaly classes for Upkie robot simulation.

This module provides various anomaly types that can be injected into
robot trajectories for testing anomaly detection methods.
"""

import numpy as np
from typing import Tuple, Any, List
from abc import ABC, abstractmethod
from upkie.utils.external_force import ExternalForce


# =============================================================================
# Helper Functions
# =============================================================================

def clear_external_forces(simulator: Any) -> None:
    simulator.set_external_forces({"torso": ExternalForce(np.zeros(3))})


# =============================================================================
# Base Class
# =============================================================================

class Anomaly(ABC):
    """Base class for anomalies."""
    
    @abstractmethod
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        """Called at the start of each anomaly episode."""
        pass
    
    @abstractmethod
    def apply(
        self,
        step: int,
        observation: np.ndarray,
        action: np.ndarray,
        simulator: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the anomaly at a given timestep.
        
        Args:
            step: Current timestep in the episode.
            observation: Current observation (may be modified).
            action: Current action (may be modified).
            simulator: The backend simulator for applying external forces.
            
        Returns:
            Modified (observation, action) tuple.
        """
        pass


class ConstantForceAnomaly(Anomaly):
    """Applies a constant external force throughout the episode."""
    
    def __init__(self, force: np.ndarray = None):
        """
        Args:
            force: 3D force vector [x, y, z]. Default is [1., 0., 0.].
        """
        self.force = force if force is not None else np.array([1.0, 0.0, 0.0])
        self.external_force = None
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        self.external_force = ExternalForce(force=self.force)
    
    def apply(self, step, observation, action, simulator):
        simulator.set_external_forces({"torso": self.external_force})
        return observation, action


class ImpulseForceAnomaly(Anomaly):
    """Applies a brief impulse force at a random time in the episode."""
    
    def __init__(
        self,
        force_magnitude: float = 5.0,
        duration_seconds: float = 0.2,
        frequency: float = 200.0,
    ):
        """
        Args:
            force_magnitude: Magnitude of the impulse force in Newtons.
            duration_seconds: How long the impulse lasts.
            frequency: Control frequency in Hz (to convert duration to steps).
        """
        self.force_magnitude = force_magnitude
        self.duration_steps = int(duration_seconds * frequency)
        self.start_step = 0
        self.force_direction = None
        self.external_force = None
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        # Random start time (not too early, not too late)
        latest_start = max(0, n_steps - self.duration_steps - 10)
        earliest_start = min(50, latest_start)
        self.start_step = rng.integers(earliest_start, max(earliest_start + 1, latest_start))
        
        # Random direction on the horizontal plane
        angle = rng.uniform(0, 2 * np.pi)
        self.force_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        force = self.force_magnitude * self.force_direction
        self.external_force = ExternalForce(force=force)
    
    def apply(self, step, observation, action, simulator):
        if self.start_step <= step < self.start_step + self.duration_steps:
            simulator.set_external_forces({"torso": self.external_force})
        else:
            clear_external_forces(simulator)
        return observation, action


class PeriodicForceAnomaly(Anomaly):
    """Applies an oscillating force (simulates vibration or wobble)."""
    
    def __init__(
        self,
        amplitude: float = 2.0,
        frequency_hz: float = 2.0,
        control_frequency: float = 200.0,
    ):
        """
        Args:
            amplitude: Peak amplitude of the oscillating force.
            frequency_hz: Oscillation frequency in Hz.
            control_frequency: Control loop frequency in Hz.
        """
        self.amplitude = amplitude
        self.omega = 2 * np.pi * frequency_hz
        self.dt = 1.0 / control_frequency
        self.phase = 0.0
        self.direction = np.array([1.0, 0.0, 0.0])
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        # Random phase offset
        self.phase = rng.uniform(0, 2 * np.pi)
        # Random direction on horizontal plane
        angle = rng.uniform(0, 2 * np.pi)
        self.direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    
    def apply(self, step, observation, action, simulator):
        t = step * self.dt
        magnitude = self.amplitude * np.sin(self.omega * t + self.phase)
        force = magnitude * self.direction
        external_force = ExternalForce(force=force)
        simulator.set_external_forces({"torso": external_force})
        return observation, action


class RandomWalkForceAnomaly(Anomaly):
    """Applies a smoothly varying random force (simulates wind gusts)."""
    
    def __init__(
        self,
        max_magnitude: float = 3.0,
        smoothness: float = 0.95,
        control_frequency: float = 200.0,
    ):
        """
        Args:
            max_magnitude: Maximum force magnitude.
            smoothness: How smoothly the force changes (0-1, higher = smoother).
            control_frequency: Control loop frequency in Hz.
        """
        self.max_magnitude = max_magnitude
        self.smoothness = smoothness
        self.current_force = np.zeros(3)
        self.rng = None
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.current_force = np.zeros(3)
    
    def apply(self, step, observation, action, simulator):
        # Random walk with smoothing
        noise = self.rng.normal(0, 1, size=3)
        noise[2] = 0  # Keep force horizontal
        self.current_force = self.smoothness * self.current_force + (1 - self.smoothness) * noise
        
        # Normalize and scale
        magnitude = np.linalg.norm(self.current_force)
        if magnitude > 1.0:
            self.current_force /= magnitude
        force = self.current_force * self.max_magnitude
        
        external_force = ExternalForce(force=force)
        simulator.set_external_forces({"torso": external_force})
        return observation, action


# =============================================================================
# Sensor Anomalies
# =============================================================================

class SensorNoiseAnomaly(Anomaly):
    """Injects noise into the observations (simulates sensor degradation)."""
    
    def __init__(self, noise_std: np.ndarray = None):
        """
        Args:
            noise_std: Standard deviation of noise for each observation channel.
                       Default is [0.05, 0.02, 0.1, 0.05] for 
                       [pitch, ground_pos, pitch_vel, ground_vel].
        """
        self.noise_std = noise_std if noise_std is not None else np.array([0.05, 0.02, 0.1, 0.05])
        self.rng = None
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        self.rng = rng
    
    def apply(self, step, observation, action, simulator):
        noise = self.rng.normal(0, self.noise_std)
        noisy_observation = observation + noise
        return noisy_observation, action


class SensorBiasAnomaly(Anomaly):
    """Adds a constant bias to observations (simulates sensor drift)."""
    
    def __init__(self, bias: np.ndarray = None, random_bias_scale: np.ndarray = None):
        """
        Args:
            bias: Fixed bias to add. If None, uses random_bias_scale.
            random_bias_scale: Scale for random bias per channel. 
                               Default is [0.1, 0.05, 0.0, 0.0].
        """
        self.fixed_bias = bias
        self.random_bias_scale = random_bias_scale if random_bias_scale is not None else np.array([0.1, 0.05, 0.0, 0.0])
        self.bias = None
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        if self.fixed_bias is not None:
            self.bias = self.fixed_bias
        else:
            self.bias = rng.uniform(-1, 1, size=len(self.random_bias_scale)) * self.random_bias_scale
    
    def apply(self, step, observation, action, simulator):
        return observation + self.bias, action


class SensorDropoutAnomaly(Anomaly):
    """Randomly drops sensor readings (replaces with zero or last value)."""
    
    def __init__(self, dropout_prob: float = 0.1, use_last_value: bool = True):
        """
        Args:
            dropout_prob: Probability of dropout at each timestep.
            use_last_value: If True, use last valid observation; if False, use zeros.
        """
        self.dropout_prob = dropout_prob
        self.use_last_value = use_last_value
        self.last_observation = None
        self.rng = None
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.last_observation = None
    
    def apply(self, step, observation, action, simulator):
        if self.rng.random() < self.dropout_prob:
            if self.use_last_value and self.last_observation is not None:
                return self.last_observation.copy(), action
            else:
                return np.zeros_like(observation), action
        else:
            self.last_observation = observation.copy()
            return observation, action


# =============================================================================
# Actuator Anomalies
# =============================================================================

class ActuatorDegradationAnomaly(Anomaly):
    """Scales down actions (simulates motor weakness)."""
    
    def __init__(self, efficiency: float = 0.5):
        """
        Args:
            efficiency: Fraction of commanded action that's actually applied (0-1).
        """
        self.efficiency = efficiency
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        pass
    
    def apply(self, step, observation, action, simulator):
        degraded_action = action * self.efficiency
        return observation, degraded_action


class ActuatorNoiseAnomaly(Anomaly):
    """Adds noise to actions (simulates motor jitter)."""
    
    def __init__(self, noise_std: float = 0.1):
        """
        Args:
            noise_std: Standard deviation of action noise.
        """
        self.noise_std = noise_std
        self.rng = None
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        self.rng = rng
    
    def apply(self, step, observation, action, simulator):
        noise = self.rng.normal(0, self.noise_std, size=action.shape)
        return observation, action + noise


class ActuatorDelayAnomaly(Anomaly):
    """Delays action application (simulates actuator lag)."""
    
    def __init__(self, delay_steps: int = 3):
        """
        Args:
            delay_steps: Number of steps to delay actions.
        """
        self.delay_steps = delay_steps
        self.action_buffer = []
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        self.action_buffer = []
    
    def apply(self, step, observation, action, simulator):
        self.action_buffer.append(action.copy())
        if len(self.action_buffer) > self.delay_steps:
            delayed_action = self.action_buffer.pop(0)
        else:
            delayed_action = np.zeros_like(action)
        return observation, delayed_action


# =============================================================================
# Timing Anomalies
# =============================================================================

class LatencyAnomaly(Anomaly):
    """Delays observations by a fixed number of steps."""
    
    def __init__(self, delay_steps: int = 5):
        """
        Args:
            delay_steps: Number of steps to delay observations.
        """
        self.delay_steps = delay_steps
        self.obs_buffer = []
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        self.obs_buffer = []
    
    def apply(self, step, observation, action, simulator):
        self.obs_buffer.append(observation.copy())
        if len(self.obs_buffer) > self.delay_steps:
            delayed_obs = self.obs_buffer.pop(0)
        else:
            delayed_obs = self.obs_buffer[0]
        return delayed_obs, action


# =============================================================================
# Composite Anomaly
# =============================================================================

class CompositeAnomaly(Anomaly):
    """Combines multiple anomalies."""
    
    def __init__(self, anomalies: List[Anomaly]):
        """
        Args:
            anomalies: List of Anomaly instances to combine.
        """
        self.anomalies = anomalies
    
    def reset(self, n_steps: int, rng: np.random.Generator) -> None:
        for anomaly in self.anomalies:
            anomaly.reset(n_steps, rng)
    
    def apply(self, step, observation, action, simulator):
        for anomaly in self.anomalies:
            observation, action = anomaly.apply(step, observation, action, simulator)
        return observation, action