"""
Task configuration dataclasses.
"""

from dataclasses import dataclass


@dataclass
class SafetyMonitorConfig:
    name: str       # environment name (for path resolution)
    win: int        # length of test windows (shorter -> harder)
    hor: int        # failure horizon (longer -> harder)
    test_size: float = 0.3  # fraction of episodes for test
    seed: int = 42  # random seed for split


# Registry mapping env name to config
TASK_CONFIGS = {
    'inv_pend': SafetyMonitorConfig('inv_pend', win=90, hor=45),
    'hopper': SafetyMonitorConfig('hopper', win=75, hor=70),
    'half_cheetah': SafetyMonitorConfig('half_cheetah', win=70, hor=10),
    'ant': SafetyMonitorConfig('ant', win=70, hor=30),
    'humanoid': SafetyMonitorConfig('humanoid', win=60, hor=30),
    'upkie': SafetyMonitorConfig('upkie', win=200, hor=100),
}
