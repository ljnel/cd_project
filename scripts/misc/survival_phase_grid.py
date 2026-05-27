"""2x3 grid of (x, y) phase plots for surviving trajectories with CD level sets.

Per env: 80% of surviving trajs (T_START onwards) colored by step + CD fit
(degree 5, Chebyshev, QR); 20% holdout -> q90 of per-trajectory max
CD value, displayed as a level set.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from data.io import load
from data.dataset import survived
from algs.cd_poly import CDPolynomial
from envs.info import ENV_INFO
from utils.paths import get_output_dir

T_START = 200
DEGREE = 6

# Per-env: extract (x, y) from sliced obs, axis labels, and termination bounds
# as (kind in {'h','v'}, value, label_or_None).
ENV_SPECS = {
    'inv_pend': dict(
        xy=lambda X: (X[..., 1], X[..., 3]),
        xlab=r'$\theta$ (rad)', ylab=r'$\dot\theta$ (rad/s)',
        bounds=[('v', -np.radians(11.5), r'$|\theta| = 11.5\degree$'),
                ('v',  np.radians(11.5), None)],
    ),
    'hopper': dict(
        xy=lambda X: (X[..., 1], X[..., 0]),
        xlab=r'$\theta$ (rad)', ylab=r'$z$ (m)',
        bounds=[('h', 0.7, r'$z = 0.7$'),
                ('v', -0.2, r'$|\theta| = 0.2$ rad'),
                ('v',  0.2, None)],
    ),
    'half_cheetah': dict(
        xy=lambda X: (X[..., 1], X[..., 10]),
        xlab=r'$\phi$ (rad)', ylab=r'$\dot\phi$ (rad/s)',
        bounds=[('v', -0.9, r'$|\phi| = 0.9$ rad'),
                ('v',  0.9, None)],
    ),
    'ant': dict(
        xy=lambda X: (np.sqrt(X[..., 2] ** 2 + X[..., 3] ** 2), X[..., 0]),
        xlab=r'$r = \sqrt{q_x^2 + q_y^2}$', ylab=r'$z$ (m)',
        bounds=[('h', 0.2, r'$z \in [0.2, 1.0]$'),
                ('h', 1.0, None),
                ('v', float(np.sin(np.radians(22.5))),
                 fr'$r = \sin(22.5\degree) \approx {np.sin(np.radians(22.5)):.3f}$')],
    ),
    'humanoid': dict(
        xy=lambda X: (X[..., 0], X[..., 24]),
        xlab=r'$z$ (m)', ylab=r'$\dot z$ (m/s)',
        bounds=[('v', 1.0, r'$z \in [1.0, 2.0]$'),
                ('v', 2.0, None)],
    ),
    'upkie': dict(
        xy=lambda X: (X[..., 0], X[..., 2]),
        xlab=r'$\theta_p$ (rad)', ylab=r'$\dot\theta_p$ (rad/s)',
        bounds=[('v', -1.0, r'$|\theta_p| = 1$ rad'),
                ('v',  1.0, None)],
    ),
}

ORDER = ['inv_pend', 'hopper', 'half_cheetah', 'ant', 'humanoid', 'upkie']


def panel(ax, env_name, ds_survived, norm):
    spec = ENV_SPECS[env_name]
    s = ds_survived.split({'fit': 0.8, 'cal': 0.2}, seed=0)
    X_fit = s['fit'].X[:, T_START:]
    X_cal = s['cal'].X[:, T_START:]
    x_fit, y_fit = spec['xy'](X_fit)
    x_cal, y_cal = spec['xy'](X_cal)
    n_fit, T = x_fit.shape

    cd = CDPolynomial(
        np.stack([x_fit.ravel(), y_fit.ravel()], axis=-1),
        degree=DEGREE, basis='cheb', method='qr',
    )
    cal2d = np.stack([x_cal.ravel(), y_cal.ravel()], axis=-1)
    cd_cal_max = np.asarray(cd(cal2d)).reshape(x_cal.shape).max(axis=1)
    q90 = float(np.quantile(cd_cal_max, 0.9))
    cond_M = float(np.linalg.cond(cd.M))

    pts = np.stack([x_fit, y_fit], axis=-1)
    segs = np.stack([pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 2, 2)
    step_color = np.tile(np.arange(T_START, T_START + T - 1), n_fit)
    alpha = float(min(0.1, 30.0 / n_fit))
    lc = LineCollection(segs, array=step_color, cmap='viridis',
                        norm=norm, lw=0.4, alpha=alpha)
    ax.add_collection(lc)

    for kind, val, label in spec['bounds']:
        fn = ax.axhline if kind == 'h' else ax.axvline
        fn(val, color='black', ls='--', lw=1.2, label=label)

    vlines = [v for k, v, _ in spec['bounds'] if k == 'v']
    hlines = [v for k, v, _ in spec['bounds'] if k == 'h']
    xs = vlines + [float(x_fit.min()), float(x_fit.max())]
    ys = hlines + [float(y_fit.min()), float(y_fit.max())]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    dx = (x_max - x_min) * 0.05 + 1e-9
    dy = (y_max - y_min) * 0.05 + 1e-9
    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    gx = np.linspace(xlim[0], xlim[1], 300)
    gy = np.linspace(ylim[0], ylim[1], 300)
    GX, GY = np.meshgrid(gx, gy)
    vals = np.asarray(cd(np.column_stack([GX.ravel(), GY.ravel()]))).reshape(GX.shape)
    ax.contour(GX, GY, vals, levels=[q90], colors='darkorange', linewidths=1.5)
    ax.plot([], [], color='darkorange', lw=1.5,
            label=fr'CD $q_{{90}}^{{\max}} = {q90:.2g}$')

    info = ENV_INFO[env_name]
    ax.set_title(f'{info.display_name}  (n_fit={n_fit}, n_cal={len(x_cal)})')
    ax.set_xlabel(spec['xlab'])
    ax.set_ylabel(spec['ylab'])
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    print(f"{env_name:>14s}: n_fit={n_fit:4d}, n_cal={len(x_cal):4d}, T={T+T_START}, "
          f"cond(M)={cond_M:.3e}, q90(max)={q90:.3g}, alpha={alpha:.3f}")
    return lc


datasets = {env: survived(load(env=env, name='base', obs_only=True)) for env in ORDER}
T_max = max(ds.X.shape[1] for ds in datasets.values())
norm = Normalize(vmin=T_START, vmax=T_max - 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
lcs = [panel(ax, env_name, datasets[env_name], norm)
       for ax, env_name in zip(axes.ravel(), ORDER, strict=True)]

fig.subplots_adjust(right=0.92, top=0.93, bottom=0.07, left=0.06, wspace=0.25, hspace=0.30)
cbar_ax = fig.add_axes((0.935, 0.07, 0.012, 0.86))
cbar = fig.colorbar(lcs[-1], cax=cbar_ax, label='step')
if cbar.solids is not None:
    cbar.solids.set(alpha=1.0)

fig.suptitle(
    f'Surviving-trajectory phase plots (steps $\\geq$ {T_START}, CD deg {DEGREE}, '
    f'$q_{{90}}$ per-traj max from 20% holdout)', y=0.98,
)
out = get_output_dir() / "survival_phase_grid.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
