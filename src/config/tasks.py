"""
Task configuration dataclasses.
"""

from dataclasses import dataclass


@dataclass
class SafetyMonitorConfig:
    name: str       # dataset name (for path resolution)
    win: int        # length of test windows (shorter -> harder)
    hor: int        # failure horizon (longer -> harder)
    test_size: float = 0.3  # fraction of episodes for test
    seed: int = 42  # random seed for split


# Example configs
inv_pend_cfg = SafetyMonitorConfig('inv_pend', win=90, hor=45)
hopper_cfg = SafetyMonitorConfig('hopper', win=75, hor=70)
half_cheetah_cfg = SafetyMonitorConfig('half_cheetah', win=70, hor=10)
humanoid_cfg = SafetyMonitorConfig('humanoid', win=60, hor=30)
upkie_cfg = SafetyMonitorConfig('upkie', win=200, hor=100)
