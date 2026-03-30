"""
Task configuration dataclasses.
"""

from dataclasses import dataclass


@dataclass
class SafetyMonitorConfig:
    name: str       # environment name (for path resolution)
    win: int        # length of test windows (shorter -> harder)
    hor: int        # failure horizon (longer -> harder)
    seed: int = 42  # random seed for split


# Registry mapping env name to config
TASK_CONFIGS = {
    'inv_pend': SafetyMonitorConfig('inv_pend', win=100, hor=30),
    'hopper': SafetyMonitorConfig('hopper', win=100, hor=80),
    'half_cheetah': SafetyMonitorConfig('half_cheetah', win=100, hor=100),
    'ant': SafetyMonitorConfig('ant', win=100, hor=20),
    'humanoid': SafetyMonitorConfig('humanoid', win=100, hor=100),
    'upkie': SafetyMonitorConfig('upkie', win=100, hor=100),
}
