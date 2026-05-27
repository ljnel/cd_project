"""Hopper (theta, z) phase plots per mass scale, with CD level set.

For each mass in MASS_SCALES, rejection-sample N_SURVIVED survived episodes (others
fixed at nominal), then plot like the hopper panel of survival_phase_grid.py.
Per-mass surviving-episode arrays are cached to scripts/misc/data/.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from algs.cd_poly import CDPolynomial
from data.generation.mujoco import _dispatch
from envs.info import ENV_INFO
from utils.paths import get_output_dir, get_root

MASS_SCALES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
N_SURVIVED = 500
MAX_RAW = 60_000  # cap raw episodes per mass (rejection sampling budget)
EP_LEN = 1000
T_START = 200
T_END = 800
DEGREE = 6

CACHE_DIR = get_root() / "outputs" / "hopper_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def collect_survived(mass: float, n_survived: int, max_raw: int = MAX_RAW) -> np.ndarray:
    """Rejection-sample up to n_survived surviving Hopper episodes at the given mass.

    Stops when n_survived is reached OR max_raw episodes have been simulated;
    returns whatever was collected (may be fewer than n_survived).
    """
    cache = CACHE_DIR / f"hopper_survived_mass{mass:.2f}_n{n_survived}.npz"
    if cache.exists():
        X = np.load(cache)['X']
        print(f"  cache hit: {cache} (n={len(X)})")
        return X

    env_info = ENV_INFO['hopper']
    gym_name = env_info.gym_name
    policy_path = str(get_root() / 'data/policies' / 'hopper-v5-sac-expert.zip')

    rng = np.random.default_rng(int(round(mass * 1000)))
    collected = []
    n_collected = 0
    n_total = 0  # episodes generated so far (for survival-rate estimate)
    seed_offset = 0

    for it in range(40):
        n_needed = n_survived - n_collected
        if n_needed <= 0 or n_total >= max_raw:
            break
        # Estimate survival rate from past iters; assume 50% if no info.
        rate = max(n_collected / n_total, 0.001) if n_total else 0.5
        n_gen = max(int(np.ceil(n_needed / rate * 1.3)), 500)
        n_gen = min(n_gen, max_raw - n_total)
        seeds = (rng.integers(0, 2**31, size=n_gen).astype(np.uint32) + seed_offset)
        seed_offset += n_gen
        mass_arr = np.full(n_gen, mass, dtype=np.float32)
        ones = np.ones(n_gen, dtype=np.float32)
        res = _dispatch(
            gym_name, policy_path, n_gen, EP_LEN,
            seeds, mass_arr, ones, ones,
            n_jobs=-1, algo='SAC',
        )
        mask = res['fail'] == EP_LEN
        collected.append(res['X'][mask])
        n_collected += int(mask.sum())
        n_total += n_gen
        print(f"  iter {it + 1}: {int(mask.sum())}/{n_gen} survived "
              f"(rate {int(mask.sum())/n_gen:.2%}), total {n_collected}/{n_survived}, "
              f"raw {n_total}/{max_raw}")

    if n_collected == 0:
        raise RuntimeError(f"no surviving episodes at mass={mass} in {n_total} raw eps")
    if n_collected < n_survived:
        print(f"  budget exhausted: collected {n_collected}/{n_survived} survived at "
              f"mass={mass} from {n_total} raw eps")

    X = np.concatenate(collected, axis=0)[:n_survived]
    np.savez(cache, X=X)
    print(f"  saved {cache} (n={len(X)})")
    return X


per_mass_X: dict[float, np.ndarray] = {}
for m in MASS_SCALES:
    print(f"\nmass = {m}")
    per_mass_X[m] = collect_survived(m, N_SURVIVED)


# ------------------------------------------------------------------ plot
norm = Normalize(vmin=T_START, vmax=T_END - 1)
n_mass = len(MASS_SCALES)
ncols = min(4, n_mass)
nrows = (n_mass + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                         sharex=True, sharey=True, squeeze=False)
flat_axes = axes.ravel().tolist()
for ax in flat_axes[n_mass:]:
    ax.set_visible(False)

lcs = []
overlays: list[dict] = []  # per-mass (GX, GY, vals, q90, lim) for overlay plot
for ax, mass in zip(flat_axes[:n_mass], MASS_SCALES, strict=True):
    X = per_mass_X[mass][:, T_START:T_END]              # (n, T, 11)
    theta = X[..., 1]
    z = X[..., 0]
    n, T = theta.shape

    # 80/20 fit/cal split for CD level set
    perm = np.random.default_rng(0).permutation(n)
    n_fit = int(0.8 * n)
    fit_idx, cal_idx = perm[:n_fit], perm[n_fit:]
    th_f, z_f = theta[fit_idx], z[fit_idx]
    th_c, z_c = theta[cal_idx], z[cal_idx]

    cd = CDPolynomial(
        np.stack([th_f.ravel(), z_f.ravel()], axis=-1),
        degree=DEGREE, basis='cheb', method='qr',
    )
    cal2d = np.stack([th_c.ravel(), z_c.ravel()], axis=-1)
    cd_cal_max = np.asarray(cd(cal2d)).reshape(th_c.shape).max(axis=1)
    q90 = float(np.quantile(cd_cal_max, 0.9))
    cond_M = float(np.linalg.cond(cd.M))

    pts = np.stack([th_f, z_f], axis=-1)
    segs = np.stack([pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 2, 2)
    step_color = np.tile(np.arange(T_START, T_START + T - 1), n_fit)
    alpha = float(min(0.1, 30.0 / n_fit))
    lc = LineCollection(segs, array=step_color, cmap='viridis',
                        norm=norm, lw=0.4, alpha=alpha)
    ax.add_collection(lc)
    lcs.append(lc)

    ax.axhline(0.7, color='black', ls='--', lw=1.2, label=r'$z = 0.7$')
    ax.axvline(-0.2, color='black', ls='--', lw=1.2, label=r'$|\theta| = 0.2$ rad')
    ax.axvline(0.2, color='black', ls='--', lw=1.2)

    x_min = min(-0.2, float(th_f.min()))
    x_max = max(0.2, float(th_f.max()))
    y_min = min(0.7, float(z_f.min()))
    y_max = float(z_f.max())
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
    overlays.append({'mass': mass, 'GX': GX, 'GY': GY, 'vals': vals, 'q90': q90,
                     'xlim': xlim, 'ylim': ylim})

    ax.set_title(fr'mass scale = {mass}  (n_fit={n_fit}, n_cal={n - n_fit})')
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$z$ (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    print(f"mass={mass}: cond(M)={cond_M:.3e}, q90(max)={q90:.3g}, alpha={alpha:.3f}")


fig.subplots_adjust(right=0.93, top=0.90, bottom=0.08, left=0.05,
                    wspace=0.18, hspace=0.28)
cbar_ax = fig.add_axes((0.945, 0.08, 0.012, 0.82))
cbar = fig.colorbar(lcs[-1], cax=cbar_ax, label='step')
if cbar.solids is not None:
    cbar.solids.set(alpha=1.0)

fig.suptitle(
    f'Hopper surviving-trajectory phase plots vs mass scale '
    f'(n={N_SURVIVED}/mass, steps {T_START}-{T_END}, CD deg {DEGREE}, '
    f'$q_{{90}}$ per-traj max from 20% holdout)',
    y=0.98,
)
out = get_output_dir() / "hopper_mass_phase_grid.png"
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")


# -------------------------------------------------------- overlay plot
fig2, ax2 = plt.subplots(figsize=(7.5, 6.5))
# Sequential single-hue shading; skip the lightest end for visibility.
cmap_m = plt.get_cmap('Blues')
overlays_sorted = sorted(overlays, key=lambda o: o['mass'])
shades = np.linspace(0.30, 0.95, len(overlays_sorted))
for o, t in zip(overlays_sorted, shades, strict=True):
    color = cmap_m(t)
    ax2.contour(o['GX'], o['GY'], o['vals'], levels=[o['q90']],
                colors=[color], linewidths=2.0)
    ax2.plot([], [], color=color, lw=2.0,
             label=fr'mass = {o["mass"]:.2f} ($q_{{90}} = {o["q90"]:.2g}$)')

ax2.axhline(0.7, color='black', ls='--', lw=1.2, label=r'$z = 0.7$')
ax2.axvline(-0.2, color='black', ls='--', lw=1.2, label=r'$|\theta| = 0.2$ rad')
ax2.axvline(0.2, color='black', ls='--', lw=1.2)

xlims = np.array([o['xlim'] for o in overlays])
ylims = np.array([o['ylim'] for o in overlays])
ax2.set_xlim(float(xlims[:, 0].min()), float(xlims[:, 1].max()))
ax2.set_ylim(float(ylims[:, 0].min()), float(ylims[:, 1].max()))

ax2.set_xlabel(r'$\theta$ (rad)')
ax2.set_ylabel(r'$z$ (m)')
ax2.set_title(
    f'Hopper CD level sets vs mass scale\n'
    f'(n={N_SURVIVED}/mass, steps {T_START}-{T_END}, CD deg {DEGREE}, '
    f'$q_{{90}}$ per-traj max from 20% holdout)'
)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
out2 = get_output_dir() / "hopper_mass_cd_overlay.png"
fig2.savefig(out2, dpi=130)
print(f"saved {out2}")
