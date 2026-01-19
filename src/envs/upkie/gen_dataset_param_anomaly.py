#!/usr/bin/env python3
"""
Test script for PyBullet parameter-based anomalies in Upkie.
"""

import numpy as np
import gymnasium as gym
import upkie.envs
import pybullet as p
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

# Register once at module level
upkie.envs.register()

OBS_DIM = 4


# =============================================================================
# Parameter-Based Anomaly Classes
# =============================================================================

class ParameterAnomaly(ABC):
    """Base class for parameter-based anomalies."""
    
    @abstractmethod
    def apply_at_reset(self, robot_id: int, rng: np.random.Generator) -> None:
        pass
    
    @abstractmethod
    def restore(self, robot_id: int) -> None:
        pass


class MassAnomaly(ParameterAnomaly):
    """Scales the mass of robot links."""
    
    def __init__(self, mass_scale: float = 1.5, link_indices: list = None):
        self.mass_scale = mass_scale
        self.link_indices = link_indices
        self.base_masses = {}
    
    def apply_at_reset(self, robot_id: int, rng: np.random.Generator) -> None:
        num_joints = p.getNumJoints(robot_id)
        indices = self.link_indices if self.link_indices else list(range(-1, num_joints))
        
        for idx in indices:
            info = p.getDynamicsInfo(robot_id, idx)
            self.base_masses[idx] = info[0]
            p.changeDynamics(robot_id, idx, mass=info[0] * self.mass_scale)
    
    def restore(self, robot_id: int) -> None:
        for idx, mass in self.base_masses.items():
            p.changeDynamics(robot_id, idx, mass=mass)
        self.base_masses = {}


class FrictionAnomaly(ParameterAnomaly):
    """Modifies friction coefficients."""
    
    def __init__(self, friction_scale: float = 0.3, link_indices: list = None):
        self.friction_scale = friction_scale
        self.link_indices = link_indices
        self.base_frictions = {}
    
    def apply_at_reset(self, robot_id: int, rng: np.random.Generator) -> None:
        num_joints = p.getNumJoints(robot_id)
        indices = self.link_indices if self.link_indices else list(range(-1, num_joints))
        
        for idx in indices:
            info = p.getDynamicsInfo(robot_id, idx)
            self.base_frictions[idx] = info[1]
            p.changeDynamics(robot_id, idx, lateralFriction=info[1] * self.friction_scale)
    
    def restore(self, robot_id: int) -> None:
        for idx, friction in self.base_frictions.items():
            p.changeDynamics(robot_id, idx, lateralFriction=friction)
        self.base_frictions = {}


class JointDampingAnomaly(ParameterAnomaly):
    """Modifies joint damping."""
    
    def __init__(self, damping_scale: float = 3.0, joint_indices: list = None):
        self.damping_scale = damping_scale
        self.joint_indices = joint_indices
        self.base_dampings = {}
    
    def apply_at_reset(self, robot_id: int, rng: np.random.Generator) -> None:
        num_joints = p.getNumJoints(robot_id)
        indices = self.joint_indices if self.joint_indices else list(range(num_joints))
        
        for idx in indices:
            joint_info = p.getJointInfo(robot_id, idx)
            self.base_dampings[idx] = joint_info[6]
            p.changeDynamics(robot_id, idx, jointDamping=joint_info[6] * self.damping_scale)
    
    def restore(self, robot_id: int) -> None:
        for idx, damping in self.base_dampings.items():
            p.changeDynamics(robot_id, idx, jointDamping=damping)
        self.base_dampings = {}


class CompositeParameterAnomaly(ParameterAnomaly):
    """Combines multiple parameter anomalies."""
    
    def __init__(self, anomalies: list):
        self.anomalies = anomalies
    
    def apply_at_reset(self, robot_id: int, rng: np.random.Generator) -> None:
        for a in self.anomalies:
            a.apply_at_reset(robot_id, rng)
    
    def restore(self, robot_id: int) -> None:
        for a in self.anomalies:
            a.restore(robot_id)


# =============================================================================
# Data Generation
# =============================================================================

def gen_data_with_param_anomaly(
    n_episodes: int = 20,
    time: float = 5.0,
    anomaly_ratio: float = 0.5,
    frequency: float = 200.0,
    anomaly: ParameterAnomaly = None,
    render: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rollout data with parameter-based anomalies."""
    
    rng = np.random.default_rng(seed)
    n_steps = int(time * frequency)
    n_anomalies = int(n_episodes * anomaly_ratio)
    anomaly_episodes = set(rng.choice(n_episodes, n_anomalies, replace=False))
    
    env = gym.make("Upkie-PyBullet-Pendulum", frequency=frequency, gui=render)
    robot_id = env.unwrapped.backend.robot_id
    
    from upkie.controllers import MPCBalancer
    
    X = np.zeros((n_episodes, n_steps, OBS_DIM), dtype=np.float32)
    y = np.zeros(n_episodes, dtype=np.float32)
    
    print(f"Generating {n_episodes} episodes ({n_anomalies} with param anomalies)")
    
    for ep in range(n_episodes):
        is_anomaly = ep in anomaly_episodes
        obs, info = env.reset()
        
        if is_anomaly and anomaly is not None:
            anomaly.apply_at_reset(robot_id, rng)
            y[ep] = 1.0
        
        mpc = MPCBalancer(
            fall_pitch=1.0, leg_length=0.58, max_ground_accel=10.0,
            max_ground_velocity=3.0, nb_timesteps=50, sampling_period=0.02
        )
        
        for step in range(n_steps):
            X[ep, step] = obs
            action = np.atleast_1d(
                mpc.compute_ground_velocity(0.0, info["spine_observation"], env.unwrapped.dt)
            ).reshape(env.action_space.shape)
            
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                X[ep, step+1:] = X[ep, step]
                break
        
        if is_anomaly and anomaly is not None:
            anomaly.restore(robot_id)
        
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{n_episodes}")
    
    env.close()
    print(f"Done: X={X.shape}, normal={int((y==0).sum())}, anomaly={int((y>0).sum())}")
    return X, y


# =============================================================================
# Visualization
# =============================================================================

def compare_plots(X, y, num_samples=3, title=""):
    normal_idx = np.where(y == 0)[0][:num_samples]
    anomaly_idx = np.where(y > 0)[0][:num_samples]
    labels = ['pitch', 'ground pos', 'ang vel', 'ground vel']
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 3), sharex=True)
    fig.suptitle(title)
    
    for i in range(num_samples):
        if i < len(normal_idx):
            axes[i, 0].plot(X[normal_idx[i]])
            axes[i, 0].set_title(f"Normal (Index {normal_idx[i]})")
        if i < len(anomaly_idx):
            axes[i, 1].plot(X[anomaly_idx[i]])
            axes[i, 1].set_title(f"Anomaly (Index {anomaly_idx[i]})")
    
    axes[0, 0].legend(labels, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()


def print_robot_structure(robot_id: int):
    """Print all links and joints of the robot."""
    num_joints = p.getNumJoints(robot_id)
    print(f"\nRobot {robot_id} has {num_joints} joints:")
    print("-" * 70)
    
    base_info = p.getDynamicsInfo(robot_id, -1)
    print(f"Base: mass={base_info[0]:.4f}, friction={base_info[1]:.4f}")
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        dyn_info = p.getDynamicsInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1].decode():30s} | "
              f"mass={dyn_info[0]:.4f}, fric={dyn_info[1]:.4f}, damp={joint_info[6]:.4f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    
    # Print robot structure
    print("=" * 60)
    print("Robot Structure")
    print("=" * 60)
    env = gym.make("Upkie-PyBullet-Pendulum", frequency=200.0, gui=False)
    env.reset()
    print_robot_structure(env.unwrapped.backend.robot_id)
    env.close()
    
    # Test anomalies
    tests = [
        ("Mass Anomaly (1.5x)", MassAnomaly(mass_scale=1.5)),
        ("Friction Anomaly (1.5x)", FrictionAnomaly(friction_scale=1.5)),
        ("Joint Damping Anomaly (4x)", JointDampingAnomaly(damping_scale=4.0)),
        ("Composite (1.4x mass + 0.3x friction)", CompositeParameterAnomaly([
            MassAnomaly(mass_scale=1.4),
            FrictionAnomaly(friction_scale=0.3),
        ])),
    ]
    
    for name, anomaly in tests:
        print(f"\n{'=' * 60}")
        print(f"Testing: {name}")
        print("=" * 60)
        
        X, y = gen_data_with_param_anomaly(
            n_episodes=10, time=5.0, anomaly_ratio=0.5,
            anomaly=anomaly, seed=42,
        )
        compare_plots(X, y, num_samples=3, title=name)
    
    print("\nAll tests complete!")