"""Grid search over cross-covariance kernel hyperparams to maximize MMD^2.

Searches over Periodic*RBF product temporal kernel and spatial kernel params.
"""

import numpy as np
import time
from itertools import product

from cd.data.datasets import load_experiment
from cd.utils.windows import strided_window_view
from cd.utils.stats import mmd_squared
from cd.algs.kernels.spatial_kernel import fit_gammas, humanoid_composite_kernel
from cd.algs.kernels.trajectory_kernels import spatiotemporal_kernel
from cd.algs.kernels.temporal_kernel import RBFKernel, PeriodicKernel

# Load data
win = 30
x_tr, x_te, y_true, _ = load_experiment('humanoid', trim=True, normalize=False, win=win)
x_tr = strided_window_view(x_tr, window=win, stride=100).reshape(-1, win, x_tr.shape[-1])

# Subsample test set: stratified 20 inliers + 20 outliers
rng = np.random.default_rng(42)
idx_in = np.where(y_true == 0)[0]
idx_out = np.where(y_true == 1)[0]
pick_in = rng.choice(idx_in, min(20, len(idx_in)), replace=False)
pick_out = rng.choice(idx_out, min(20, len(idx_out)), replace=False)
pick = np.concatenate([pick_in, pick_out])
x_sub = x_te[pick]
y_sub = y_true[pick]
print(f"Test subset: {len(x_sub)} ({(y_sub==0).sum()} inliers, {(y_sub==1).sum()} outliers)")

# Fit spatial gammas from training data
gammas = fit_gammas(x_tr)
print(f"Fitted gammas: {gammas}")

# Use best spatial params from previous search
gamma_r = 60.0
gamma_z = 0.1

spatial_kw = {**gammas, 'gamma_r': gamma_r, 'gamma_z': gamma_z}
spatial_fn = lambda x1, x2: humanoid_composite_kernel(x1, x2, **spatial_kw)

# Temporal grid: Periodic * RBF product kernel
t_grid = np.linspace(0, 1, win)
ls_rbf_values = [0.1, 0.2, 0.5, 1.0, 2.0]
ls_per_values = [0.05, 0.1, 0.2, 0.5, 1.0]
period_values = [0.1, 0.15, 0.2, 0.25, 0.33]

# Also include pure RBF baselines for comparison
configs = []

# Pure RBF baselines
for ls in ls_rbf_values:
    K_t = RBFKernel(length_scale=ls)(t_grid)
    configs.append(('rbf', {'ls_rbf': ls}, K_t))

# Periodic * RBF product kernels
for ls_rbf, ls_per, period in product(ls_rbf_values, ls_per_values, period_values):
    k = PeriodicKernel(length_scale=ls_per, period=period) * RBFKernel(length_scale=ls_rbf)
    K_t = k(t_grid)
    configs.append(('per*rbf', {'ls_rbf': ls_rbf, 'ls_per': ls_per, 'period': period}, K_t))

total = len(configs)
print(f"\nRunning {total} configurations...")

results = []
t0 = time.time()

for i, (name, params, K_t) in enumerate(configs):
    K = spatiotemporal_kernel(x_sub, K_t=K_t, spatial_kernel_fn=spatial_fn)
    score = mmd_squared(K, y_sub)
    results.append({'kernel': name, 'mmd2': score, **params})

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (total - i - 1) / rate
        print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

# Sort and display top results
results.sort(key=lambda r: r['mmd2'], reverse=True)

print(f"\n{'rank':>4} {'mmd2':>10} {'kernel':>8} {'ls_rbf':>8} {'ls_per':>8} {'period':>8}")
print("-" * 52)
for i, r in enumerate(results[:25]):
    print(f"{i+1:>4} {r['mmd2']:>10.4f} {r['kernel']:>8} {r['ls_rbf']:>8.2f} {r.get('ls_per', '-'):>8} {r.get('period', '-'):>8}")
