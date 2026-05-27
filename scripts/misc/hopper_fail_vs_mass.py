"""Throw-away: failure rate vs mass scale for Hopper (mass in [0.5, 1.5])."""
import numpy as np
import matplotlib.pyplot as plt

from data.generation.mujoco import _dispatch
from envs.info import ENV_INFO
from utils.paths import get_root

env_info = ENV_INFO['hopper']
gym_name = env_info.gym_name
policy_path = str(get_root() / 'data/policies' / 'hopper-v5-sac-expert.zip')

N_MASS = 11
N_PER_MASS = 50
EP_LEN = 1000

mass_grid = np.linspace(0.5, 1.5, N_MASS).astype(np.float32)
mass_scale = np.repeat(mass_grid, N_PER_MASS)
n_eps = len(mass_scale)
friction_scale = np.ones(n_eps, dtype=np.float32)
damping_scale = np.ones(n_eps, dtype=np.float32)
seeds = np.arange(n_eps, dtype=np.uint32)

print(f"Running {n_eps} episodes ({N_MASS} mass values x {N_PER_MASS} eps), ep_len={EP_LEN}...")
result = _dispatch(
    gym_name, policy_path, n_eps, EP_LEN,
    seeds, mass_scale, friction_scale, damping_scale,
    n_jobs=-1, algo='SAC',
)

fail = result['fail']
failed = fail < EP_LEN

rates = np.array([failed[mass_scale == m].mean() for m in mass_grid])
mean_fail_step = np.array([
    fail[(mass_scale == m) & failed].mean() if failed[mass_scale == m].any() else np.nan
    for m in mass_grid
])

print("\nmass_scale | fail_rate | mean fail step (failed only)")
for m, r, fs in zip(mass_grid, rates, mean_fail_step):
    fs_s = f"{fs:6.1f}" if not np.isnan(fs) else "  n/a "
    print(f"  {m:5.2f}    |  {r:6.1%}  |  {fs_s}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(mass_grid, rates, 'o-', lw=2)
ax.set_xlabel('mass scale')
ax.set_ylabel('failure rate')
ax.set_title(f'Hopper failure rate vs mass scale (n={N_PER_MASS}/mass, ep_len={EP_LEN})')
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = 'scripts/misc/hopper_fail_vs_mass.png'
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
