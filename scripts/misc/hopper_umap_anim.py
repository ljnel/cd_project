#!/usr/bin/env python3
import warnings
import webbrowser
from typing import Literal

import numpy as np
import tyro

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import umap

from data.io import load
from envs.info import ENV_INFO
from utils.paths import get_output_dir, get_root


def render_video(env, dataset, video, fps, dpi, ttf_vmax,
                 chosen, fails, cens, Z_bg, bg_ttf, frame_steps, snap):
    """Render the same scene as the HTML to an MP4/GIF via matplotlib. The
    static layers (background scatter, trajectory lines, start dots) are drawn
    once; only the moving markers and the appearing end-markers update per
    frame. Markers are drawn last (highest zorder) so they sit on top.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    vmax, cmap = ttf_vmax, 'magma_r'
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    pad = 0.03
    xmin, xmax = float(Z_bg[:, 0].min()), float(Z_bg[:, 0].max())
    ymin, ymax = float(Z_bg[:, 1].min()), float(Z_bg[:, 1].max())
    ax.set_xlim(xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin))
    ax.set_ylim(ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin))
    ax.autoscale(False)

    # Static layers (drawn once).
    ax.scatter(Z_bg[:, 0], Z_bg[:, 1], c=bg_ttf, cmap=cmap, vmin=0, vmax=vmax,
               s=3, alpha=0.2, linewidths=0, rasterized=True, zorder=0)
    for e in chosen:
        ax.plot(e['Z'][:, 0], e['Z'][:, 1], color='0.47', alpha=0.2, lw=0.8,
                zorder=1)
    ax.scatter([e['Z'][0, 0] for e in chosen], [e['Z'][0, 1] for e in chosen],
               s=28, c='lime', edgecolors='black', linewidths=0.7, zorder=3,
               label='start')

    # Dynamic artists.
    empty = np.empty((0, 2))
    fail_sc = ax.scatter([], [], s=70, marker='x', c='red', linewidths=1.5,
                         zorder=4, label='failure')
    cens_sc = ax.scatter([], [], s=55, marker='s', facecolors='none',
                         edgecolors='dimgray', linewidths=1.4, zorder=4,
                         label='censored')
    mark_sc = ax.scatter([], [], s=70, c=[], cmap=cmap, vmin=0, vmax=vmax,
                         edgecolors='black', linewidths=1.0, zorder=5)

    sm = ScalarMappable(norm=Normalize(0, vmax), cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label('TTF (clipped)')
    ax.set_title(f"{ENV_INFO[env].display_name} {dataset}: "
                 f"UMAP trajectories of {len(chosen)} episodes")
    ax.legend(loc='upper left', framealpha=0.6)
    step_txt = ax.text(0.99, 0.99, '', transform=ax.transAxes, va='top',
                       ha='right', fontsize=10)

    def update(t):
        alive = [e for e in chosen if t < len(e['Z']) - 1]
        s = [snap(e, t) for e in alive]
        mark_sc.set_offsets([(p[0], p[1]) for p in s] if s else empty)
        mark_sc.set_array(np.array([p[2] for p in s]))
        ef = [e for e in fails if t >= len(e['Z']) - 1]
        fail_sc.set_offsets([(e['Z'][-1, 0], e['Z'][-1, 1]) for e in ef]
                            if ef else empty)
        ec = [e for e in cens if t >= len(e['Z']) - 1]
        cens_sc.set_offsets([(e['Z'][-1, 0], e['Z'][-1, 1]) for e in ec]
                            if ec else empty)
        step_txt.set_text(f"sim step: {t}")
        return mark_sc, fail_sc, cens_sc, step_txt

    anim = FuncAnimation(fig, update, frames=frame_steps, blit=False)
    out = get_output_dir() / f"hopper_umap_anim.{video}"
    writer = (FFMpegWriter(fps=fps, bitrate=3000) if video == 'mp4'
              else PillowWriter(fps=fps))
    print(f"rendering {len(frame_steps)} frames to {video} at "
          f"{fps} fps (this can take a minute)")
    anim.save(out, writer=writer, dpi=dpi,
              progress_callback=lambda i, n: (i % 50 == 0 or i == n - 1)
              and print(f"  frame {i + 1}/{n}"))
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main(
    env: str = 'hopper',
    dataset: str = 'base',
    max_step: int = 5000,
    bg_size: int = 30_000,
    n_episodes: int = 50,
    stride: int = 5,
    frame_ms: int = 40,
    video: Literal['mp4', 'gif'] | None = None,
    fps: int = 30,
    dpi: int = 130,
    ttf_vmax: float = 2000,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    seed: int = 42,
):
    """Animated UMAP of hopper state trajectories (interactive plotly HTML).

    Fits UMAP on a background subsample of all hopper/base states (including the
    rolled-out continuations past T from scripts/misc/hopper_umap_ttf.py's cache),
    then projects 50 randomly chosen episodes through that fit and renders animated
    scatter: each chosen episode is drawn as a faint static trajectory line, and a
    larger marker per episode moves through its UMAP coordinates as the slider
    scrubs through global time. Marker color encodes time-to-failure (same scale
    as the static plot).

    Args:
        env: Environment name.
        dataset: Dataset name.
        max_step: Global step at which the rolled-out continuation stops.
        bg_size: Background subsample size for UMAP fit + scatter.
        n_episodes: Episodes to animate (uniformly random).
        stride: Frame stride in sim steps; plotly tweens between frames so
            playback stays smooth.
        frame_ms: Per-frame duration in ms during playback; also the tween
            duration (smaller = faster motion).
        video: Render a static MP4/GIF (for slides) instead of the interactive
            HTML.
        fps: Frames per second for --video output.
        dpi: Resolution for --video output.
        ttf_vmax: TTF value mapped to the light end of the colormap.
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
        seed: RNG seed.
    """
    ds = load(env, dataset)
    X, fail = ds.X, ds.fail
    N, T, _ = X.shape
    horizon = max_step - (T - 1)

    cache = get_root() / 'outputs' / 'hopper_umap_ttf' / 'continuations.npz'
    if not cache.exists():
        raise SystemExit(
            f"missing {cache}; run scripts/misc/hopper_umap_ttf.py first to "
            f"populate the continuation cache.")
    npz = np.load(cache, allow_pickle=True)
    trajs = list(npz['trajs'])
    ttfs = npz['ttfs']

    surv_idx = np.flatnonzero(fail == T)
    surv_pos = {int(i): j for j, i in enumerate(surv_idx)}
    true_fail = fail.astype(np.int64).copy()
    true_fail[surv_idx] = (T - 1) + ttfs

    eps = []
    for i in range(N):
        if fail[i] < T:
            obs = X[i, :int(fail[i]) + 1]
            censored = False
        else:
            j = surv_pos[i]
            if ttfs[j] == horizon:
                obs = X[i, :T]
                censored = True
            else:
                obs = np.concatenate([X[i, :T], trajs[j]], axis=0)
                censored = False
        valid = ~np.isnan(obs).any(axis=1)
        obs = obs[valid].astype(np.float32)
        if len(obs) == 0:
            continue
        eps.append({'i': i, 'obs': obs,
                    'ttf': true_fail[i] - np.arange(len(obs)),
                    'true_fail': int(true_fail[i]),
                    'censored': censored})

    all_X = np.concatenate([e['obs'] for e in eps], axis=0)
    all_ttf = np.concatenate([e['ttf'] for e in eps], axis=0)
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(all_X), size=min(bg_size, len(all_X)),
                        replace=False)
    bg_X = all_X[bg_idx]
    bg_ttf = all_ttf[bg_idx]
    print(f"fitting UMAP on {len(bg_X)} background states "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist})")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=seed).fit(bg_X)
    Z_bg = reducer.embedding_

    n_pick = min(n_episodes, len(eps))
    sel = rng.choice(len(eps), size=n_pick, replace=False)
    chosen = [eps[int(k)] for k in sel]
    print(f"selected {len(chosen)} episodes uniformly at random")

    n_proj_states = sum(len(e['obs']) for e in chosen)
    print(f"projecting {n_proj_states} states through UMAP")
    for e in chosen:
        e['Z'] = reducer.transform(e['obs'])

    T_max = max(len(e['obs']) for e in chosen)
    frame_steps = list(range(0, T_max, stride))
    if frame_steps[-1] != T_max - 1:
        frame_steps.append(T_max - 1)  # land on last index so every cross shows
    print(f"building animation: {len(frame_steps)} frames, "
          f"{len(chosen)} markers/frame")

    def snap(e, t):
        idx = min(t, len(e['Z']) - 1)
        return float(e['Z'][idx, 0]), float(e['Z'][idx, 1]), int(e['ttf'][idx])

    bg_layer = go.Scattergl(
        x=Z_bg[:, 0], y=Z_bg[:, 1],
        mode='markers',
        marker=dict(size=2, color=bg_ttf, cmin=0, cmax=ttf_vmax,
                    colorscale='magma_r', opacity=0.2),
        name='all states (subsample)', hoverinfo='skip',
    )
    traj_layers = []
    for e in chosen:
        traj_layers.append(go.Scattergl(
            x=e['Z'][:, 0], y=e['Z'][:, 1],
            mode='lines',
            line=dict(color='rgba(120,120,120,0.20)', width=1),
            name=f"ep {e['i']} (true_fail={e['true_fail']}"
                 + (", censored" if e['censored'] else "") + ")",
            hoverinfo='skip', showlegend=False,
        ))

    start_xs = [float(e['Z'][0, 0]) for e in chosen]
    start_ys = [float(e['Z'][0, 1]) for e in chosen]
    start_layer = go.Scattergl(
        x=start_xs, y=start_ys, mode='markers',
        marker=dict(size=6, color='lime', symbol='circle',
                    line=dict(width=1, color='black')),
        text=[f"start ep {e['i']}" for e in chosen],
        hoverinfo='text', name='start', showlegend=True,
    )

    fails = [e for e in chosen if not e['censored']]
    cens = [e for e in chosen if e['censored']]

    if video:
        render_video(env, dataset, video, fps, dpi, ttf_vmax,
                     chosen, fails, cens, Z_bg, bg_ttf, frame_steps, snap)
        return

    # The three animated traces use SVG go.Scatter (few points each) so plotly
    # can update just them per frame with redraw=False; the heavy static layers
    # (30k-point bg + trajectory lines) stay Scattergl and are drawn once.
    #
    # Every trace keeps a FIXED point order (one slot per episode), with hidden
    # slots set to None. Stable per-index identity is what lets plotly tween
    # marker positions between frames — otherwise points would glide to the
    # wrong targets as episodes appear/disappear.
    def end_trace(subset, t, *, symbol, color, line_w, name, label):
        on = [t >= len(e['Z']) - 1 for e in subset]  # episode has ended by t
        return go.Scatter(
            x=[float(e['Z'][-1, 0]) if k else None for e, k in zip(subset, on)],
            y=[float(e['Z'][-1, 1]) if k else None for e, k in zip(subset, on)],
            mode='markers',
            marker=dict(size=9, color=color, symbol=symbol,
                        line=dict(width=line_w, color='black' if line_w == 1
                                  else color)),
            text=[f"{label} ep {e['i']} (t={len(e['Z']) - 1})" for e in subset],
            hoverinfo='text', name=name, showlegend=True,
        )

    def fail_trace(t):
        return end_trace(fails, t, symbol='x', color='red', line_w=1,
                         name='failure', label='failure')

    def cens_trace(t):
        return end_trace(cens, t, symbol='square-open', color='dimgray',
                         line_w=1.5, name='censored (no failure)',
                         label='censored')

    def marker_trace(t, *, colorbar=False):
        # Live while t < last index; once ended the slot goes None (vanishes)
        # and the matching end-marker takes over.
        alive = [t < len(e['Z']) - 1 for e in chosen]
        snaps = [snap(e, t) for e in chosen]
        marker = dict(size=10, color=[s[2] for s in snaps],
                      cmin=0, cmax=ttf_vmax, colorscale='magma_r',
                      line=dict(width=1.2, color='black'))
        if colorbar:
            marker['colorbar'] = dict(title='TTF<br>(clipped)')
        return go.Scatter(
            x=[s[0] if a else None for s, a in zip(snaps, alive)],
            y=[s[1] if a else None for s, a in zip(snaps, alive)],
            mode='markers', marker=marker,
            text=[f"ep {e['i']} | t={t} | true_fail={e['true_fail']}"
                  for e in chosen],
            hoverinfo='text', name='current', showlegend=True,
        )

    fail_layer = fail_trace(0)
    cens_layer = cens_trace(0)
    marker_layer = marker_trace(0, colorbar=True)

    # Trajectory lines go at the very bottom so the TTF-colored background
    # scatter (and the animated markers) aren't occluded by the grey paths.
    layers = (traj_layers
              + [bg_layer, start_layer, fail_layer, cens_layer, marker_layer])
    marker_idx = len(layers) - 1
    cens_idx = len(layers) - 2
    fail_idx = len(layers) - 3

    frames = []
    for t in frame_steps:
        frames.append(go.Frame(
            data=[fail_trace(t), cens_trace(t), marker_trace(t)],
            traces=[fail_idx, cens_idx, marker_idx],
            name=str(t),
        ))

    # The slider re-renders every tick on each playback frame, so one step per
    # frame (thousands) throttles playback. Give it a coarse subsample of steps
    # for scrubbing; the play button still advances through every frame.
    n_slider = min(len(frame_steps), 150)
    slider_ts = [frame_steps[k] for k in
                 sorted(set(np.linspace(0, len(frame_steps) - 1, n_slider)
                            .round().astype(int).tolist()))]

    fig = go.Figure(data=layers, frames=frames)
    fig.update_layout(
        title=dict(
            text=f"{ENV_INFO[env].display_name} {dataset}: "
                 f"UMAP trajectories of {len(chosen)} episodes "
                 f"(stride={stride})",
            x=0.5, xanchor='center', y=0.985, yanchor='top'),
        xaxis=dict(visible=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(visible=False),
        width=1000, height=860,
        margin=dict(l=20, r=90, t=100, b=90),
        # Horizontal legend along the top so it doesn't collide with the
        # colorbar pinned to the right edge.
        legend=dict(orientation='h', x=0.5, xanchor='center',
                    y=1.0, yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.6)'),
        sliders=[{
            'active': 0,
            'x': 0.12, 'len': 0.88, 'xanchor': 'left',
            'y': 0, 'yanchor': 'top', 'pad': {'b': 10, 't': 50},
            'steps': [{'method': 'animate',
                       'args': [[str(t)], {'mode': 'immediate',
                                           'frame': {'duration': frame_ms,
                                                     'redraw': False},
                                           'transition': {'duration': 0}}],
                       'label': str(t)} for t in slider_ts],
            'currentvalue': {'prefix': 'sim step: '},
        }],
        updatemenus=[{
            'type': 'buttons',
            'direction': 'left',
            'showactive': False,
            'x': 0.1, 'xanchor': 'right',
            'y': 0, 'yanchor': 'top', 'pad': {'r': 10, 't': 45},
            'buttons': [
                {'label': '▶ play',
                 'method': 'animate',
                 # duration:0 = no hold between frames; the motion is the tween
                 # itself, so each transition chains directly into the next.
                 'args': [None, {'frame': {'duration': 0, 'redraw': False},
                                 'fromcurrent': True,
                                 'mode': 'immediate',
                                 'transition': {'duration': frame_ms,
                                                'easing': 'linear'}}]},
                {'label': '⏸ pause',
                 'method': 'animate',
                 'args': [[None], {'mode': 'immediate',
                                   'frame': {'duration': 0, 'redraw': False},
                                   'transition': {'duration': 0}}]},
            ],
        }],
    )

    out = get_output_dir() / 'hopper_umap_anim.html'
    fig.write_html(out, include_plotlyjs='cdn', full_html=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    webbrowser.open(out.as_uri())


if __name__ == '__main__':
    tyro.cli(main)
