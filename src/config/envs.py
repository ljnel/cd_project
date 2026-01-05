from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    "Wrapper that includes env name, policy file name, params for domain randomization"
    name: str
    policy: str
    dof_damping: Tuple[float, float]
    mass: Tuple[float, float]
    fric: Tuple[float, float]


ENV_CFG = {
    'inv_pend': EnvConfig(
        name="InvertedPendulum-v5",
        policy="invertedpendulum-v5-sac-expert.zip",
        dof_damping=(0.6, 20.0),
        mass=(0.6, 2.4),
        fric=(0.6, 2.4),
    ),
    'hopper': EnvConfig(
        name="Hopper-v5",
        policy="hopper-v5-sac-expert.zip",
        dof_damping=(1.0, 1.0),
        mass=(1.0, 1.0),
        fric=(1.0, 1.0),
    ),
    'half_cheetah': EnvConfig(
        name="HalfCheetah-v5",
        policy="halfcheetah-v5-sac-expert.zip",
        dof_damping=(0.4, 2.8),
        mass=(0.4, 2.2),
        fric=(0.4, 2.2)
    ),
    'ant': EnvConfig(
        name="Ant-v5",
        policy="ant-v5-sac-expert.zip",
        dof_damping=(0.6, 2.8),
        mass=(0.6, 1.8),
        fric=(0.6, 1.4)
    ),
    'humanoid': EnvConfig(
        name="Humanoid-v5",
        policy="humanoid-v5-sac-expert.zip",
        dof_damping=(0.9, 1.1),
        mass=(0.9, 1.1),
        fric=(0.9, 1.1)
    )

}
