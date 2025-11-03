import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser
from pathlib import Path
from utils.misc import time_call

import glob
from sklearn.metrics import precision_recall_curve, auc
from algs.cd_poly import CDPolynomial
from algs.state_window_ae import StateSeqAutoencoder  # or wherever you place it
from algs.conv_ae import Conv1dStateSeqAutoencoder
from algs.pair_ae import PairAE
import lightning as L
from torch.utils.data import DataLoader
from sklearn.random_projection import GaussianRandomProjection
from algs.pair_ae import PairAE

np.random.seed(0)

n_eps = 1000 # should be deduced from data
ep_len = 1000 # should be deduced from data
q = 0.99 # quantile for computing thresholds
skip = 20

"""
Run this file to benchmark a method.
Example:    python benchmark.py --env hopper --method dist --plot
"""

def process_data(file: str, s_dim, a_dim):
    npz = np.load(file)
    ds = {
        's': npz['states'],
        'a': npz['actions'],
        'r': npz['rew'],
        'fail': npz['fail'],
        'params': npz['params'],
        'seeds': npz['seeds']
    }

    # do some data preprocessing
    # step-level features -> reshape
    ds['s'] = ds['s'][:, ::skip]
    ds['a'] = ds['a'][:, ::skip]
    ds['r'] = ds['r'][:, ::skip]
    # episode-level features -> tile
    ds['fail_step'] = ds['fail'] / skip # in [0, n_steps]
    ds['fail'] = ds['fail'] > 0

    ds['t'] = np.arange(0, ep_len, step=skip) / ep_len
    
    # print some statistics...
    print(f'State stats: {ds["s"].min():.3f}, {ds["s"].max():.3f}')
    print(f'Action stats: {ds["a"].min():.3f}, {ds["a"].max():.3f}')

    return ds

def mask_ds(ds: dict, mask):
    """Return a shallow copy of ds with episode-level arrays masked."""
    masked = {}
    for k, v in ds.items():
        # Only mask episode-major arrays
        if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0]:
            masked[k] = v[mask]
        else:
            masked[k] = v
    return masked


def get_ts(ds: dict):
    "Returns step-level time-state data."
    n_eps = ds['s'].shape[0] # if masked
    S = ds['s'].reshape(n_eps * n_steps, -1)
    if ae is not None:
        S = ae.encode(torch.Tensor(S))
    
    X =  np.concatenate([
        np.tile(ds['t'], n_eps)[:, np.newaxis],
        S
    ] , axis=1)
    print(f'State stats: {S.min()}, {S.max()}')    
    return X

def get_tpss(ds: dict):
    """
    Returns step-level [time, prev_state, state] data.
    At the first step of each episode, prev_state = state.
    """
    n_eps, n_steps, s_dim = ds['s'].shape
    s = ds['s']
    # prev = shift s one step right, pad first with itself
    prev_s = np.concatenate([s[:, :1], s[:, :-1]], axis=1)
    t = np.tile(ds['t'], n_eps)[:, None]
    #return np.concatenate([t, prev_s.reshape(-1, s_dim), s.reshape(-1, s_dim)], axis=1)
    return np.concatenate([prev_s.reshape(-1, s_dim), s.reshape(-1, s_dim)], axis=1)

""" def get_sa(ds: dict):
    "Returns step-level state-action data."
    n_eps = ds['s'].shape[0] # if masked
    return np.concatenate([
        ds['s'].reshape(n_eps * n_steps, -1),
        ds['a'].reshape(n_eps * n_steps, -1)
    ] , axis=1) """

def get_sa(ds: dict, step: int):
    """Return state-action features for all episodes at a given step."""
    s = ds['s'][:, step, :]   # (n_eps, s_dim)
    a = ds['a'][:, step, :]   # (n_eps, a_dim)
    return np.concatenate([s, a], axis=1)


#---------- Methods to benchmark --------------------#

"""
Methods should take as input feature matrices X_tr, X_cal, X_te
should return
"""

def cd_poly(ds_tr: dict, cal_mask, ds_te, args):
    fail = ds_tr['fail']

    X_tr_succ = get_ts(mask_ds(ds_tr, ~fail & ~cal_mask))
    X_cal = get_ts(mask_ds(ds_tr, ~fail & cal_mask))
    print(f'Calibration set size: {(~fail & cal_mask).sum()}')
    X_te = get_ts(ds_te)

    p = CDPolynomial(X_tr_succ, degree=args.deg, verbose=True, basis='cheb', eps=0)

    scores_cal = np.log(p(X_cal)).reshape(-1, n_steps)
    scores_te = np.log(p(X_te)).reshape(-1, n_steps)

    t = time_call(lambda X: np.log(p(X)), X_te)

    #thresh = np.quantile(scaores_cal, q)  # step-level calibration
    thresh = np.quantile(scores_cal.max(axis=1), q)  # episode-level calibration

    return scores_te, np.full(n_steps, thresh), t


def distance_based(ds_tr, cal_mask, ds_te, args):
    from algs.distance_baseline import DistanceBasedOneClass

    fail = ds_tr['fail']
    tr_succ = mask_ds(ds_tr, ~fail & ~cal_mask)
    cal_succ = mask_ds(ds_tr, ~fail & cal_mask)

    clfs = []
    for step in range(n_steps):
        clf = DistanceBasedOneClass()
        X_tr_step = get_sa(tr_succ, step)   # (n_tr_step, feat_dim)
        clf.fit(X_tr_step)
        clfs.append(clf)

    scores_te = np.zeros((n_eps, n_steps))
    total_time = 0.0

    # Score test set
    for step, clf in enumerate(clfs):
        X_te_step = get_sa(ds_te, step)    # (n_eps, feat_dim)
        t = time_call(clf.score_samples, X_te_step)
        total_time += t
        scores_te[:, step] = clf.score_samples(X_te_step)

    # Calibrate thresholds per step
    """ thresh = np.zeros(n_steps)
    for step, clf in enumerate(clfs):
        X_cal_step = get_sa(cal_succ, step)  # (n_cal_eps, feat_dim)
        cal_scores = clf.score_samples(X_cal_step)
        thresh[step] = np.quantile(cal_scores, q) """

    # --- Calibration scores ---
    cal_scores = np.zeros((cal_succ['s'].shape[0], n_steps))
    for step, clf in enumerate(clfs):
        X_cal_step = get_sa(cal_succ, step)  # (n_cal_eps, feat_dim)
        cal_scores[:, step] = clf.score_samples(X_cal_step)

    # --- Threshold: quantile of episode-max scores ---
    cal_episode_max = cal_scores.max(axis=1)     # (n_cal_eps,)
    thresh = np.quantile(cal_episode_max, q)

    return scores_te, np.full(n_steps, thresh), total_time


def cd_rp(ds_tr: dict, cal_mask, ds_te: dict, args):
    """
    Ensemble of Random Projections + CDPolynomial with k-of-m voting.
    """

    fail = ds_tr['fail']
    X_tr_succ = get_ts(mask_ds(ds_tr, ~fail & ~cal_mask))
    X_cal     = get_ts(mask_ds(ds_tr, ~fail &  cal_mask))
    X_te      = get_ts(ds_te)

    ensemble = args.ensemble
    proj_dim = args.proj
    #k = max(1, int(0.2 * ensemble))  # default 20% voting
    k = 2

    scores_te_all, proj_thresh = [], []
    total_time = 0.0

    for seed in range(ensemble):
        proj = GaussianRandomProjection(n_components=proj_dim, random_state=seed)

        tr_proj = np.concatenate([X_tr_succ[:, :1], proj.fit_transform(X_tr_succ[:, 1:])], axis=1)
        cal_proj = np.concatenate([X_cal[:, :1], proj.transform(X_cal[:, 1:])], axis=1)
        te_proj  = np.concatenate([X_te[:, :1],  proj.transform(X_te[:, 1:])],  axis=1)

        p = CDPolynomial(tr_proj, degree=args.deg, verbose=True, basis='cheb', eps=0)

        scores_cal = np.log(p(cal_proj)).reshape(-1, n_steps)
        scores_te  = np.log(p(te_proj)).reshape(-1, n_steps)

        total_time += time_call(lambda X: np.log(p(X)), te_proj)

        scores_te_all.append(scores_te)

        tau = np.quantile(scores_cal.max(axis=1), q)
        proj_thresh.append(tau)

    votes = [(S >= tau) for S, tau in zip(scores_te_all, proj_thresh)]  # list of bools
    votes = np.stack(votes, axis=0)  # (m, n_eps, n_steps)

    pred = (votes.sum(axis=0) >= k)          # k-of-m voting
    pred = np.maximum.accumulate(pred, 1)    # cumulative over time

    return pred, total_time



def ae_recon(df_tr, df_cal, df_te, args):
    assert ae.hparams.no_fail
    x_tr = torch.Tensor(get_sa(df_tr).astype(np.float32))
    with torch.no_grad():
        x_tr_hat, _ = ae(x_tr)
    df_tr['score'] = ((x_tr_hat - x_tr) ** 2).mean(dim=1).detach().numpy()

    def get_scores():
        x_te = torch.Tensor(get_sa(df_te).astype(np.float32))
        with torch.no_grad():
            x_te_hat, _ = ae(x_te)
        return ((x_te_hat - x_te) ** 2).mean(dim=1).detach().numpy()
    
    t = time_call(get_scores)
    df_te['score'] = get_scores()

    x_cal = torch.Tensor(get_sa(df_cal).astype(np.float32))
    with torch.no_grad():
        x_cal_hat, _ = ae(x_cal)
    df_cal['score'] = ((x_cal_hat - x_cal) ** 2).mean(dim=1).detach().numpy()
    #thresh = df_cal[~df_cal['fail']]['score'].max()
    thresh = np.quantile(df_cal[~df_cal['fail']]['score'], q)

    return np.full(n_steps, thresh), t


def log_reg(ds_tr, cal_mask, ds_te, args):
    """ A simple baseline: logistic regression on time-state features. """
    from sklearn.linear_model import LogisticRegression
    X_tr = get_ts(ds_tr)
    y_tr = np.repeat(ds_tr['fail'], n_steps).astype(int)
    model = LogisticRegression().fit(X_tr, y_tr)
    scores = model.predict_proba(get_ts(ds_te))[:, 1].reshape(n_eps, n_steps)
    print(scores.shape)
    return scores, np.full(n_steps, 0.5), 0.
    

def state_seq_recon(df_tr, df_cal, df_te, args):
    # load trained GRU model
    ckpt_dir = Path(f'./{args.env}/conv/lightning_logs/{args.enc}/checkpoints')
    ae_file = glob.glob(str(ckpt_dir/'*.ckpt'))[-1]
    model = Conv1dStateSeqAutoencoder.load_from_checkpoint(ae_file).to('cpu')
    model.eval()

    window_len = model.hparams.window_len
    state_cols = [f's{i}' for i in range(s_dim)]  # only states

    def compute_scores(df):
        scores = np.zeros(len(df))
        for ep in df['ep'].unique():
            ep_states = df[df['ep'] == ep][state_cols].to_numpy().astype(np.float32)
            ep_scores = np.zeros(len(ep_states))
            # slide window
            for i in range(len(ep_states) - window_len + 1):
                window = torch.tensor(ep_states[i:i+window_len]).unsqueeze(0)
                with torch.no_grad():
                    recon = model(window).squeeze(0).numpy()
                mse = ((recon - ep_states[i:i+window_len])**2).mean()
                ep_scores[i+window_len-1] = mse  # assign to last step of window
            scores[df['ep'] == ep] = ep_scores
        return scores

    df_tr['score'] = compute_scores(df_tr)
    df_cal['score'] = compute_scores(df_cal)
    t = time_call(lambda: compute_scores(df_te))
    df_te['score'] = compute_scores(df_te)

    # threshold from calibration successes
    thresh = np.quantile(df_cal[~df_cal['fail']]['score'], q)
    return np.full(n_steps, thresh), t


""" def state_seq_cd(df_tr, df_cal, df_te, args):
    # ---- load trained GRU state-sequence autoencoder ----
    ckpt_dir = Path(f'./{args.env}/state_window/lightning_logs/{args.enc}/checkpoints')
    ae_file = glob.glob(str(ckpt_dir/'*.ckpt'))[-1]
    model = StateSeqAutoencoder.load_from_checkpoint(ae_file).to('cpu')
    model.eval()

    window_len = model.hparams.window_len
    state_cols = [f's{i}' for i in range(s_dim)]

    # ---- helper: make all sliding windows in one array ----
    def make_windows(ep_states, window_len):
        n = len(ep_states) - window_len + 1
        return np.stack([ep_states[i:i+window_len] for i in range(n)], axis=0)

    # ---- helper: encode windows in batch ----
    def encode_windows(model, ep_states, window_len):
        windows = make_windows(ep_states, window_len)   # (n_win, L, state_dim)
        windows = torch.tensor(windows, dtype=torch.float32)
        with torch.no_grad():
            _, h = model.encoder(windows)               # h: (1, n_win, hidden_dim)
            z = model.fc_enc(h[-1])                     # (n_win, latent_dim)
        return z.numpy()                                # np array

    # ---- fit CD polynomial on training successes ----
    Z = []
    for ep in df_tr[~df_tr['fail']]['ep'].unique():
        ep_df = df_tr[df_tr['ep'] == ep]
        ep_states = ep_df[state_cols].to_numpy().astype(np.float32)
        if len(ep_states) < window_len:
            continue
        Z.append(encode_windows(model, ep_states, window_len))
    X_succ = np.vstack(Z)
    print(X_succ.shape)
    p = CDPolynomial(X_succ, degree=args.deg, verbose=True, basis='cheb', eps=0)

    # ---- scoring function ----
    def score_df(df):
        scores = np.zeros(len(df))
        for ep in df['ep'].unique():
            ep_df = df[df['ep'] == ep]
            ep_states = ep_df[state_cols].to_numpy().astype(np.float32)

            if len(ep_states) < window_len:
                continue

            # batch encode all windows
            z_all = encode_windows(model, ep_states, window_len)  # (n_win, latent_dim)
            s_all = np.log(p(z_all)).astype(float).ravel()        # (n_win,)

            ep_scores = np.zeros(len(ep_states))
            ep_scores[window_len-1:] = s_all

            # fill early steps with first valid score
            if window_len > 1:
                ep_scores[:window_len-1] = ep_scores[window_len-1]

            scores[df['ep'] == ep] = ep_scores
        return scores

    # ---- assign scores ----
    df_tr['score'] = score_df(df_tr)
    df_cal['score'] = score_df(df_cal)
    t = time_call(lambda: score_df(df_te))
    df_te['score'] = score_df(df_te)

    # ---- threshold from calibration successes ----
    thresh = np.quantile(df_cal[~df_cal['fail']]['score'], q)
    return np.full(n_steps, thresh), t """

""" def state_seq_cd(ds_tr: dict, cal_mask, ds_te: dict, args):
    # ---- load trained Conv1d model ----
    ckpt_dir = Path(f'./{args.env}/conv/lightning_logs/{args.enc}/checkpoints')
    ae_file = glob.glob(str(ckpt_dir/'*.ckpt'))[-1]
    model = Conv1dStateSeqAutoencoder.load_from_checkpoint(ae_file).to('cpu')
    model.eval()

    window_len = model.hparams.window_len
    # shapes
    n_eps_tr, n_steps_tr, sdim_tr = ds_tr['s'].shape
    n_eps_te, n_steps_te, sdim_te = ds_te['s'].shape
    assert sdim_tr == s_dim == sdim_te, "state_dim mismatch"

    # helpers
    def make_windows(ep_states, L):
        n = len(ep_states) - L + 1
        if n <= 0:
            return None
        return np.stack([ep_states[i:i+L] for i in range(n)], axis=0)  # (n_win, L, s_dim)

    @torch.no_grad()
    def encode_windows(ep_states):
        ws = make_windows(ep_states, window_len)
        if ws is None:
            return None
        ws_t = torch.tensor(ws, dtype=torch.float32)  # (n_win, L, s_dim)
        z = model.encode(ws_t)                        # (n_win, latent_dim)
        return z.numpy()

    # ---- collect latents for train successes (split: train/cal) ----
    fail = ds_tr['fail']
    tr_mask  = (~fail) & (~cal_mask)
    cal_mask_ = (~fail) & (cal_mask)

    def collect_latents(ds, mask):
        Z = []
        for ep_idx in np.where(mask)[0]:
            ep_states = ds['s'][ep_idx].astype(np.float32)  # (n_steps, s_dim)
            z = encode_windows(ep_states)
            if z is not None and len(z) > 0:
                Z.append(z)
        if len(Z) == 0:
            return np.empty((0, model.hparams.latent_dim), dtype=np.float32)
        return np.vstack(Z)

    X_tr_succ = collect_latents(ds_tr, tr_mask)      # (N_tr_win, lat_dim)
    X_cal     = collect_latents(ds_tr, cal_mask_)    # (N_cal_win, lat_dim)

    # ---- fit CD poly on training successes ----
    p = CDPolynomial(X_tr_succ, degree=args.deg, verbose=True, basis='cheb', eps=0)

    # ---- score test episodes, aggregate window scores to step level ----
    def score_dataset(ds):
        scores = np.zeros((ds['s'].shape[0], ds['s'].shape[1]), dtype=np.float32)  # (n_eps, n_steps)
        for ep_idx in range(ds['s'].shape[0]):
            ep_states = ds['s'][ep_idx].astype(np.float32)  # (n_steps, s_dim)
            z = encode_windows(ep_states)
            if z is None:
                # no full window: leave zeros
                continue
            s_all = np.log(p(z)).astype(np.float32).ravel()  # (n_win,)
            # assign to last timestep of each window
            scores[ep_idx, window_len-1:] = s_all
            # fill early steps with first valid score for continuity
            if window_len > 1:
                scores[ep_idx, :window_len-1] = s_all[0]
        return scores

    # calibration and test scores
    cal_scores = score_dataset(mask_ds(ds_tr, cal_mask_))
    get_scores_te = lambda: score_dataset(ds_te)
    t = time_call(get_scores_te)
    scores_te = get_scores_te()

    # episode-level threshold from calibration successes
    # take per-episode max over time, then q-quantile
    cal_ep_max = cal_scores.max(axis=1) if cal_scores.size else np.array([np.inf], dtype=np.float32)
    thresh = np.quantile(cal_ep_max, q) if cal_scores.size else np.inf

    return scores_te, np.full(n_steps, thresh), t """


def state_seq_cd(ds_tr: dict,
                 cal_mask: np.ndarray,
                 ds_te: dict,
                 args,
                 window_stride: int = 1,
                 batch_size: int = 8192,
                 chunk_size: int = 200_000):
    """
    CD on latent windows from a Conv1d state-sequence AE.
    Faster: batches window encoding; safe with skip>1; no hard-coded n_steps.
    Returns:
        scores_te: (n_eps_te, n_steps_te) step-level scores
        thresh:    (n_steps_te,) per-step threshold (episode-level quantile, broadcasted)
        t:         time to score the *test* set latents (seconds)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load trained Conv1d model (robust to BN/interp/deconv changes) ----
    ckpt_dir = Path(f'./{args.env}/conv/lightning_logs/{args.enc}/checkpoints')
    ae_file = glob.glob(str(ckpt_dir / '*.ckpt'))[-1]
    model = Conv1dStateSeqAutoencoder.load_from_checkpoint(
        ae_file,
        strict=False,        # tolerate tiny name/shape diffs (e.g., BN toggle)
        map_location='cpu',
    ).eval()

    L = int(model.hparams.window_len)

    # ---- Shapes (derived from downsampled arrays) ----
    n_eps_tr, n_steps_tr, sdim_tr = ds_tr['s'].shape
    n_eps_te, n_steps_te, sdim_te = ds_te['s'].shape
    assert sdim_tr == sdim_te, "state_dim mismatch between train and test"

    # ---- Helpers -----------------------------------------------------------

    @torch.no_grad()
    def encode_latents_batched(ds: dict, episode_mask: np.ndarray):
        """
        Collect all windows (with stride) across masked episodes,
        encode in big batches, and return:
            Z_all: (N_total_windows, latent_dim)
            meta:  list of (ep_idx, first_assigned_step, n_windows_for_ep)
        """
        ep_idxs = np.where(episode_mask)[0]
        W_list = []
        meta = []
        for ep in ep_idxs:
            S = ds['s'][ep].astype(np.float32)  # (T, C), already downsampled
            T, C = S.shape
            if T < L:
                meta.append((ep, 0, 0))
                continue
            W = sliding_window_view(S, (L, C))[:, 0, :]   # (T-L+1, L, C)
            if window_stride > 1:
                W = W[::window_stride]
            n_win = len(W)
            if n_win == 0:
                meta.append((ep, 0, 0))
                continue
            W_list.append(W)
            meta.append((ep, L - 1, n_win))  # scores assigned at ends of windows

        if not W_list:
            lat_dim = int(getattr(model.hparams, "latent_dim", 0) or model.latent_dim)
            return np.empty((0, lat_dim), dtype=np.float32), meta

        W_all = np.vstack(W_list)  # (sum_n_win, L, C)

        # encode in batches
        model.to(device).eval()
        Z_chunks = []
        for start in range(0, len(W_all), batch_size):
            end = min(start + batch_size, len(W_all))
            w_batch = torch.from_numpy(W_all[start:end]).to(device, non_blocking=True)
            z = model.encode(w_batch).cpu().numpy()  # (B, latent_dim)
            Z_chunks.append(z.astype(np.float32))
        Z_all = np.vstack(Z_chunks) if Z_chunks else np.empty((0, 0), dtype=np.float32)
        return Z_all, meta

    def eval_log_p_big(p, X: np.ndarray) -> np.ndarray:
        """Chunked evaluation of log p(X) to limit RAM."""
        if X.size == 0:
            return np.empty((0,), dtype=np.float32)
        out = np.empty((len(X),), dtype=np.float32)
        for start in range(0, len(X), chunk_size):
            end = min(start + chunk_size, len(X))
            out[start:end] = np.log(p(X[start:end])).astype(np.float32).ravel()
        return out

    def scatter_episode_max(cal_scores_flat: np.ndarray, meta: list) -> np.ndarray:
        """Gather per-episode max over windows using meta cursor pointers."""
        vals = []
        cursor = 0
        for _, _, n_win in meta:
            if n_win == 0:
                continue
            s_ep = cal_scores_flat[cursor:cursor + n_win]
            vals.append(s_ep.max())
            cursor += n_win
        return np.array(vals, dtype=np.float32) if vals else np.empty((0,), dtype=np.float32)

    def scatter_to_timesteps(flat_scores: np.ndarray, meta: list, n_eps: int, n_steps: int) -> np.ndarray:
        """Scatter each window score to the *last* timestep of its window per episode."""
        scores = np.zeros((n_eps, n_steps), dtype=np.float32)
        cursor = 0
        for ep_idx, base_k, n_win in meta:
            if n_win == 0:
                continue
            s_ep = flat_scores[cursor:cursor + n_win]
            cursor += n_win
            assign = base_k + np.arange(n_win) * window_stride
            assign = assign[assign < n_steps]
            if len(assign) == 0:
                continue
            s_ep = s_ep[:len(assign)]
            scores[ep_idx, assign] = s_ep
            # prefix fill for continuity
            first = int(assign[0])
            if L > 1 and first > 0:
                scores[ep_idx, :first] = s_ep[0]
        return scores

    # ---- Collect latents: train successes (split into train/cal) ------------
    fail = ds_tr['fail'].astype(bool)
    tr_mask = (~fail) & (~cal_mask)
    cal_mask_succ = (~fail) & (cal_mask)

    X_tr_succ, _        = encode_latents_batched(ds_tr, tr_mask)
    X_cal_lat, cal_meta = encode_latents_batched(ds_tr, cal_mask_succ)

    # ---- Fit CD polynomial on training successes ---------------------------
    p = CDPolynomial(X_tr_succ, degree=args.deg, verbose=True, basis='cheb', eps=0)

    # ---- Calibration threshold (episode-level, q-quantile of per-ep max) ----
    if X_cal_lat.size:
        cal_scores_flat = eval_log_p_big(p, X_cal_lat)
        cal_ep_max = scatter_episode_max(cal_scores_flat, cal_meta)
        thresh_val = np.quantile(cal_ep_max, q) if cal_ep_max.size else np.inf
    else:
        thresh_val = np.inf

    # ---- Test scoring (timed) -----------------------------------------------
    def score_test_once():
        X_te_lat, te_meta = encode_latents_batched(ds_te, np.ones(n_eps_te, dtype=bool))
        flat_scores = eval_log_p_big(p, X_te_lat)
        return scatter_to_timesteps(flat_scores, te_meta, n_eps_te, n_steps_te)

    t = time_call(score_test_once)
    scores_te = score_test_once()

    # Per-step threshold broadcast (we calibrated at episode level)
    thresh = np.full(n_steps_te, thresh_val, dtype=np.float32)
    return scores_te, thresh, t


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default='hopper')
    parser.add_argument("--method", type=str, default='cd')
    parser.add_argument("--enc")
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--proj", type=int)
    parser.add_argument("--deg", type=int, default=3)
    parser.add_argument("--plot", action='store_true')
    #parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--video", action='store_true')
    args = parser.parse_args()

    #----- get the env -----#
    if args.env == 'inv_pend':
        s_dim, a_dim = 4, 1
    if args.env == 'hopper':
        s_dim, a_dim = 11, 3
    if args.env == 'half_cheetah':
        s_dim, a_dim = 17, 6
    if args.env == 'ant':
        s_dim, a_dim = 105, 8
    if args.env == 'humanoid':
        s_dim, a_dim = 348, 17
    dir = Path(f'./{args.env}')

    #skip = args.skip # should divide ep_len
    assert ep_len % skip == 0
    n_steps = ep_len // skip

    if args.enc is not None and args.method != 'state_seq_cd':
        ckpt_dir = dir/'ae'/'lightning_logs'/args.enc/'checkpoints'
        ae_file = glob.glob(str(ckpt_dir/'*.ckpt'))[-1]
        ae = PairAE.load_from_checkpoint(ae_file).to('cpu')
        ae.eval()

        no_fail = ae.hparams.no_fail
        hid_dim = ae.hparams.hidden_dim
        lat_dim = ae.hparams.latent_dim
    else:
        ae = None
        no_fail = None
        hid_dim = None
        lat_dim = None

    """ def get_tsa(df):
        if ae is not None and args.method == 'cd_ae':
            z = ae.encode(torch.Tensor(df[sa_cols].to_numpy()))
            return np.concatenate([df['t'].to_numpy()[:,np.newaxis], 
                                   z.detach().numpy()], 
                                   axis=1)
        else:
            return df[['t'] + sa_cols].to_numpy()
    
    def get_sa(df):
        return df[sa_cols].to_numpy() """


    #------- process data --------#
    ds_tr = process_data(dir/'train.npz', s_dim, a_dim)
    ds_te = process_data(dir/'test.npz', s_dim, a_dim)

    #eps = df_tr['ep'].unique()
    n_cal_eps = max(1, int(0.2 * n_eps))
    cal_idx = np.random.randint(n_eps, size=n_cal_eps)
    cal_mask = np.zeros(n_eps, dtype=bool)
    cal_mask[cal_idx] = True

    #------- fit and predict -------#
    assert args.method in ['cd', 'cd_ae', 'dist', 'ae_rec', 'cd_rp', 'log_reg', 'state_seq_ae', 'state_seq_cd']
    if args.method in ['cd', 'cd_ae']:
        scores, thresh, t = cd_poly(ds_tr, cal_mask, ds_te, args)
    elif args.method == 'dist':
        scores, thresh, t = distance_based(ds_tr, cal_mask, ds_te, args)
    elif args.method == 'ae_rec':
        scores, thresh, t = ae_recon(ds_tr, cal_mask, ds_te, args)
    elif args.method == 'cd_rp':
        pred, t = cd_rp(ds_tr, cal_mask, ds_te, args)
    elif args.method == 'log_reg':
        scores, thresh, t = log_reg(ds_tr, cal_mask, ds_te, args)
    elif args.method == 'state_seq_ae':
        scores, thresh, t = state_seq_recon(ds_tr, cal_mask, ds_te, args)
    elif args.method == 'state_seq_cd':
        scores, thresh, t = state_seq_cd(ds_tr, cal_mask, ds_te, args)

    if args.method != 'cd_rp':
        pred = scores >= thresh  # scores: (n_eps, n_steps); thresh: (n_steps,)  

    #------- compute metrics --------#

    # --- Episode-level masks ---
    fail_eps = ds_te['fail']        # shape (n_eps,), True if episode failed
    pred_eps = pred.any(axis=1)     # shape (n_eps,), True if anomaly flagged

    tpr = pred_eps[fail_eps].mean()               # True positive rate (recall on failing eps)
    tnr = (~pred_eps[~fail_eps]).mean()           # True negative rate (specificity on success eps)
    bal_acc = 0.5 * (tpr + tnr)
    prop_fail = fail_eps.mean()
    weight_acc = prop_fail * tpr + (1 - prop_fail) * tnr

    # --- First detection time per episode ---
    flag_step = np.argmax(pred, axis=1)          # (n_eps,) index of first True
    flagged = pred.any(axis=1)
    flag_step[~flagged] = n_steps               # use sentinel = "no detection"

    # --- Fail step per episode ---
    fail_step = ds_te['fail_step']                     # (n_eps,) failure step index (downsampled)

    # --- Time-to-detect (only on failing eps with detection) ---
    mask_detected = fail_eps & flagged
    ttd = (fail_step[mask_detected] - flag_step[mask_detected]).mean() * skip 
    edr = np.mean(flag_step[mask_detected] < fail_step[mask_detected])

    #---------- Save results -------------#
    import json
    logfile = 'results.json'
    results = {
        'method': args.method,
        'env': args.env,
        'enc': args.enc,
        #'no_fail': no_fail,
        #'hid': hid_dim,  # should be deduced from env?
        #'lat': lat_dim,
        'proj': args.proj,
        'deg': args.deg,
        'tpr': tpr,
        'tnr': tnr,
        'bal_acc': bal_acc,
        'weight_acc': weight_acc,
        'ttd': ttd, # in raw env steps
        'edr': edr,
        't': t
    }
    print(results)
    with open(logfile, 'a') as f:
        f.write(json.dumps(results) + '\n')

    #if args.plot:
    #    make_plot(df_te, thresh)

    if args.plot:
        plt.plot(scores[fail_eps].T, color='red', alpha=0.1)
        plt.plot(scores[~fail_eps].T, color='green', alpha=0.1)
        plt.plot(thresh, color='blue')
        plt.title(f'{args.method} on {args.env}')
        plt.xlabel('Step')
        plt.ylabel('Anomaly score')
        plt.show()


    # --------- Full-episode video with red flash on detection + score plot ----------
    """ if args.video:
        try:
            import gymnasium as gym
            import imageio.v2 as imageio
            from pathlib import Path
            from PIL import Image, ImageDraw
            import os

            # --- Choose a representative failing episode ---
            mask_detected = fail_eps & flagged
            ep_pick = None
            if mask_detected.any():
                lead_k = (fail_step - flag_step)[mask_detected]
                fail_k = fail_step[mask_detected]
                lead_ratio = lead_k / fail_k
                target = np.quantile(lead_ratio, 0.9)
                ep_pick = np.where(mask_detected)[0][np.argmin(np.abs(lead_ratio - target))]
            else:
                # fallback: first failing episode
                failing_eps = np.where(fail_eps)[0]
                if len(failing_eps) > 0:
                    ep_pick = failing_eps[0]

            if ep_pick is None:
                raise RuntimeError("No failing episodes to visualize.")

            print(f"[video] Using representative episode {ep_pick}")
            print(f'flag step: {flag_step[ep_pick] * skip}, fail step: {fail_step[ep_pick] * skip}')

            env_map = {
                'hopper': 'Hopper-v5',
                'half_cheetah': 'HalfCheetah-v5',
                'ant': 'Ant-v5',
                'humanoid': 'Humanoid-v5',
                'inv_pend': 'InvertedPendulum-v5',
            }
            env_id = env_map.get(args.env, args.env)

            raw_te = np.load(dir/'test.npz')
            A = raw_te['actions'][ep_pick]  # (ep_len, a_dim)
            seed = int(ds_te['seeds'][ep_pick])

            ep_scores = scores[ep_pick]         # (n_steps,)
            ep_thresh = thresh                  # broadcasted (n_steps,)
            ep_preds  = pred[ep_pick]           # (n_steps,)

            # first crossing
            first_cross_k = np.argmax(ep_preds) if ep_preds.any() else None
            first_cross_t = first_cross_k * skip if first_cross_k is not None else None
            fail_env_step = int(fail_step[ep_pick] * skip)

            # upsample scores to env resolution
            score_env = np.repeat(ep_scores, skip)[:ep_len]
            thresh_env = np.repeat(ep_thresh, skip)[:ep_len]
            pred_env = (score_env >= thresh_env)

            # --- video writer ---
            os.makedirs('videos', exist_ok=True)
            try:
                import imageio_ffmpeg  # noqa: F401
                out_path = Path('videos') / f'{args.env}_{args.method}_ep{ep_pick}_allsteps.mp4'
                writer = imageio.get_writer(out_path, format='FFMPEG', fps=30, codec='libx264')
            except Exception:
                out_path = Path('videos') / f'{args.env}_{args.method}_ep{ep_pick}_allsteps.gif'
                writer = imageio.get_writer(out_path, format='GIF', duration=1/30)

            # helpers
            def flash_color(frame_np, color, alpha=0.6):
                if color is None:
                    return frame_np
                overlay = np.zeros_like(frame_np)
                overlay[..., :3] = color
                return (alpha * overlay + (1 - alpha) * frame_np).astype(np.uint8)

            def annotate_step(frame_np, t, score, thresh):
                im = Image.fromarray(frame_np)
                draw = ImageDraw.Draw(im)
                msg = f"t={t}   score={score:.3f}   threshold={thresh:.3f}"
                xy = (12, 12)
                draw.text((xy[0]+1, xy[1]+1), msg, fill=(0,0,0))
                draw.text(xy, msg, fill=(255,255,255))
                return np.asarray(im)

            env = gym.make(env_id, render_mode='rgb_array')
            obs, _ = env.reset(seed=seed)

            m = env.unwrapped.model
            m.dof_damping[:] *= ds_te['params'][ep_pick, 0]
            m.body_mass[:] *= ds_te['params'][ep_pick, 1]
            m.geom_friction[:] *= ds_te['params'][ep_pick, 2]            

            orange_start_t = first_cross_t if first_cross_t is not None else None

            for t in range(ep_len):
                obs, rew, term, trunc, info = env.step(A[t])
                frame = env.render()

                # flash coloring
                color = None
                if t >= fail_env_step:
                    color = (255, 0, 0)   # red after failure
                elif (orange_start_t is not None) and t >= orange_start_t:
                    color = (255, 165, 0) # orange after detection

                frame = flash_color(frame, color)
                frame = annotate_step(frame, t, float(score_env[t]), float(thresh_env[t]))
                writer.append_data(frame)

            writer.close()
            env.close()
            print(f"[video] Saved to {out_path}")

            # --- plot scores ---
            plt.figure()
            x = np.arange(ep_len)
            plt.plot(x, score_env, label='score')
            plt.plot(x, thresh_env, label='threshold', linestyle=':', color='black')
            if first_cross_t is not None:
                plt.axvline(first_cross_t, color='orange', linestyle='--', label='anomaly detected')
            plt.axvline(fail_env_step, color='red', linestyle='--', label='failure')

            plt.xlabel('Step')
            plt.ylabel('Score')
            plt.title(f'Anomaly score for {args.method} on {args.env} — episode {ep_pick}')
            if first_cross_t is not None:
                plt.axvspan(first_cross_t, fail_env_step, color='orange', alpha=0.3)
            plt.axvspan(fail_env_step, ep_len, color='red', alpha=0.3)
            plt.legend()
            plot_path = Path('videos') / f'{args.env}_{args.method}_ep{ep_pick}_score.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[plot] Saved to {plot_path}")

        except Exception as e:
            print(f"[video] Skipping video/plot due to error: {e}") """


    # --------- Full-episode video with red flash on detection + score plot ----------
if args.video:
    try:
        import gymnasium as gym
        import imageio.v2 as imageio
        from pathlib import Path
        from PIL import Image, ImageDraw
        import os
        import numpy as np  # (in case it's not already imported up top)

        # --- Choose a representative failing episode ---
        mask_detected = fail_eps & flagged
        ep_pick = None
        if mask_detected.any():
            lead_k = (fail_step - flag_step)[mask_detected]
            fail_k = fail_step[mask_detected]
            lead_ratio = lead_k / fail_k
            target = np.quantile(lead_ratio, 0.9)
            ep_pick = np.where(mask_detected)[0][np.argmin(np.abs(lead_ratio - target))]
        else:
            # fallback: first failing episode
            failing_eps = np.where(fail_eps)[0]
            if len(failing_eps) > 0:
                ep_pick = failing_eps[0]

        if ep_pick is None:
            raise RuntimeError("No failing episodes to visualize.")

        print(f"[video] Using representative episode {ep_pick}")
        print(f'flag step: {flag_step[ep_pick] * skip}, fail step: {fail_step[ep_pick] * skip}')

        env_map = {
            'hopper': 'Hopper-v5',
            'half_cheetah': 'HalfCheetah-v5',
            'ant': 'Ant-v5',
            'humanoid': 'Humanoid-v5',
            'inv_pend': 'InvertedPendulum-v5',
        }
        env_id = env_map.get(args.env, args.env)

        raw_te = np.load(dir/'test.npz')
        A = raw_te['actions'][ep_pick]  # (ep_len, a_dim)
        seed = int(ds_te['seeds'][ep_pick])

        ep_scores = scores[ep_pick]         # (n_steps,)
        ep_thresh = thresh                  # broadcasted (n_steps,)
        ep_preds  = pred[ep_pick]           # (n_steps,)

        # first crossing
        first_cross_k = np.argmax(ep_preds) if ep_preds.any() else None
        first_cross_t = first_cross_k * skip if first_cross_k is not None else None
        fail_env_step = int(fail_step[ep_pick] * skip)

        # upsample scores to env resolution
        score_env = np.repeat(ep_scores, skip)[:ep_len]
        thresh_env = np.repeat(ep_thresh, skip)[:ep_len]
        pred_env = (score_env >= thresh_env)

        # --- video writer ---
        os.makedirs('videos', exist_ok=True)
        try:
            import imageio_ffmpeg  # noqa: F401
            out_path = Path('videos') / f'{args.env}_{args.method}_ep{ep_pick}_allsteps.mp4'
            writer = imageio.get_writer(out_path, format='FFMPEG', fps=30, codec='libx264')
        except Exception:
            out_path = Path('videos') / f'{args.env}_{args.method}_ep{ep_pick}_allsteps.gif'
            writer = imageio.get_writer(out_path, format='GIF', duration=1/30)

        # helpers
        def flash_color(frame_np, color, alpha=0.6):
            if color is None:
                return frame_np
            overlay = np.zeros_like(frame_np)
            overlay[..., :3] = color
            return (alpha * overlay + (1 - alpha) * frame_np).astype(np.uint8)

        def annotate_step(frame_np, t, score, thresh):
            im = Image.fromarray(frame_np)
            draw = ImageDraw.Draw(im)
            msg = f"t={t}   score={score:.3f}   threshold={thresh:.3f}"
            xy = (12, 12)
            draw.text((xy[0]+1, xy[1]+1), msg, fill=(0,0,0))
            draw.text(xy, msg, fill=(255,255,255))
            return np.asarray(im)

        # --- NEW: small label helper for keyframe captions ---
        def add_caption(frame_np, caption):
            im = Image.fromarray(frame_np)
            draw = ImageDraw.Draw(im)
            pad = 8
            # draw a subtle caption bar at the bottom
            w, h = im.size
            bar_h = 28
            draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0, 127))
            draw.text((pad, h - bar_h + 6), caption, fill=(255, 255, 255))
            return np.asarray(im)

        env = gym.make(env_id, render_mode='rgb_array')
        obs, _ = env.reset(seed=seed)

        m = env.unwrapped.model
        m.dof_damping[:] *= ds_te['params'][ep_pick, 0]
        m.body_mass[:] *= ds_te['params'][ep_pick, 1]
        m.geom_friction[:] *= ds_te['params'][ep_pick, 2]            

        orange_start_t = first_cross_t if first_cross_t is not None else None

        # --- NEW: decide which steps to capture for the PNG ---
        t_begin = min(10, ep_len - 1)  # "near the beginning"
        t_orange = orange_start_t
        t_red = fail_env_step

        # --- NEW: placeholders for key frames (raw render for 'begin', flashed for others) ---
        begin_frame = None
        orange_frame = None
        red_frame = None

        for t in range(ep_len):
            obs, rew, term, trunc, info = env.step(A[t])
            frame_raw = env.render()  # keep an unmodified copy for the 'begin' shot

            # capture the raw early frame
            if begin_frame is None and t == t_begin:
                # annotate step but do NOT flash; then add a caption
                tmp = annotate_step(frame_raw, t, float(score_env[t]), float(thresh_env[t]))
                begin_frame = add_caption(tmp, "start (early episode)")

            # flash coloring for video + for orange/red keyframes
            color = None
            if t >= fail_env_step:
                color = (255, 0, 0)   # red after failure
            elif (orange_start_t is not None) and t >= orange_start_t:
                color = (255, 165, 0) # orange after detection

            frame = flash_color(frame_raw, color)
            frame = annotate_step(frame, t, float(score_env[t]), float(thresh_env[t]))

            # capture the first orange frame exactly when it starts
            if orange_frame is None and (t_orange is not None) and t == t_orange:
                orange_frame = add_caption(frame, "anomaly detected (orange starts)")

            # capture the first red frame exactly when it starts
            if red_frame is None and t == t_red:
                red_frame = add_caption(frame, "failure (red starts)")

            writer.append_data(frame)

        writer.close()
        env.close()
        print(f"[video] Saved to {out_path}")

        # --- NEW: robust fallbacks if some frames were not captured ---
        # If no detection happened, create a neutral "no detection" middle frame
        if orange_frame is None:
            # use the frame just before failure if possible, otherwise reuse begin
            t_mid = max(0, t_red - 1) if t_red is not None else t_begin
            # Re-render a best-effort neutral middle from what we annotated in video:
            # fall back to begin_frame clone with an updated caption
            if begin_frame is not None:
                im = Image.fromarray(begin_frame).copy()
                orange_frame = add_caption(np.asarray(im), "no anomaly detected")
            else:
                orange_frame = begin_frame  # may still be None; handled below

        # If any are still None, duplicate a neighbor so the PNG is always written
        if begin_frame is None:
            begin_frame = orange_frame if orange_frame is not None else red_frame
        if orange_frame is None:
            orange_frame = begin_frame if begin_frame is not None else red_frame
        if red_frame is None:
            red_frame = orange_frame if orange_frame is not None else begin_frame

        # --- NEW: compose and save the 3-frame PNG ---
        try:
            im1 = Image.fromarray(begin_frame)
            im2 = Image.fromarray(orange_frame)
            im3 = Image.fromarray(red_frame)

            # resize to a common size (use the first as reference)
            w, h = im1.size
            im2 = im2.resize((w, h))
            im3 = im3.resize((w, h))

            triptych = Image.new("RGB", (w * 3, h))
            triptych.paste(im1, (0, 0))
            triptych.paste(im2, (w, 0))
            triptych.paste(im3, (2 * w, 0))

            png_path = Path('videos') / f'{args.env}_{args.method}_ep{ep_pick}_keyframes.png'
            triptych.save(png_path, format="PNG", compress_level=6, optimize=True)
            print(f"[png] Saved keyframes to {png_path}")
        except Exception as e:
            print(f"[png] Failed to save keyframes PNG: {e}")

    except Exception as e:
        print("[video] Error:", e)
