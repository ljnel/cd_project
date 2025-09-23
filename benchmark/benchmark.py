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
from algs.pair_ae import PairAE
import lightning as L
from torch.utils.data import DataLoader
from sklearn.random_projection import GaussianRandomProjection
from algs.pair_ae import PairAE

np.random.seed(0)

n_eps = 1000
n_steps = 50
skip = 20

q = 0.99 # quantile for computing thresholds

"""
Run this file to benchmark a method.
Example:    python benchmark.py --env hopper --method dist --plot
"""

def process_data(file: str, s_dim, a_dim):
    npz = np.load(file)
    sa = npz["sa"]
    r = npz["rew"]
    param = npz["param"]
    fail = npz['fail']

    cols = [f"s{i}" for i in range(s_dim)] + [f"a{i}" for i in range(a_dim)]
    df = pd.DataFrame(
        data=sa,
        columns=cols
    )
    df['step'] = np.array([[i for i in range(n_steps)] for j in range(n_eps)]).flatten()
    df['t'] = df['step'] / n_steps # not in seconds
    df['ep'] = np.array([[i] * n_steps for i in range(n_eps)]).flatten()
    df['r'] = r
    df['damp'] = np.concatenate([np.full((n_steps,), param[i]) for i in range(n_eps)])

    df["cum_r"] = df.groupby('ep')['r'].cumsum()

    df['fail'] = (fail[df['ep']] > 0).astype(bool)
    df['fail_step'] = np.concatenate([[fail[i]] * n_steps for i in range(n_eps)])

    #df[cols] = np.sin(df[cols])
    #df['s1'] = np.sin(df['s1'])

    return df


def make_plot(df_te, thresh):
    for i in df_te[~df_te['fail']]['ep'].unique():
        plt.plot(range(n_steps), df_te[~df_te['fail']][df_te[~df_te['fail']]['ep'] == i]['score'], color='green', alpha=0.2)
    for i in df_te[df_te['fail']]['ep'].unique():
        plt.plot(range(n_steps), df_te[df_te['fail']][df_te[df_te['fail']]['ep'] == i]['score'], color='red', alpha=0.2)
    plt.plot(thresh, color='blue')
    plt.show()

#---------- Methods to benchmark --------------------#

def cd_poly(df_tr, df_cal, df_te, args):
    print(get_tsa(df_tr[~df_tr['fail']]).min(),
          get_tsa(df_tr[~df_tr['fail']]).max())

    p = CDPolynomial(get_tsa(df_tr[~df_tr['fail']]), degree=args.deg, verbose=True, basis='cheb', eps=0)
    df_tr['score'] = np.log(p(get_tsa(df_tr)))

    t = time_call(lambda X: np.log(p(get_tsa(X))), df_te)
    df_te['score'] = np.log(p(get_tsa(df_te)))

    df_cal['score'] = np.log(p(get_tsa(df_cal)))
    #thresh = df_cal[~df_cal['fail']]['score'].max()
    thresh = np.quantile(df_cal[~df_cal['fail']]['score'], q)

    return np.full(n_steps, thresh), t


def distance_based(df_tr, df_cal, df_te, args):
    from algs.distance_baseline import DistanceBasedOneClass

    clfs = []
    for step in range(n_steps):
        clf = DistanceBasedOneClass()
        X_tr_step = get_sa(df_tr[(~df_tr['fail']) & (df_tr['step'] == step)])
        clf.fit(X_tr_step)
        clfs.append(clf)

    X_te_by_step = [ get_sa(df_te[df_te['step'] == k]) for k in range(n_steps) ]
    total_time = 0.
    for step, clf in enumerate(clfs):
        X = X_te_by_step[step]
        t = time_call(clf.score_samples, X)
        total_time += t
        df_te.loc[df_te['step'] == step, 'score'] = clf.score_samples(X)

    thresh = np.zeros(n_steps)
    for step, clf in enumerate(clfs):
        X_cal_step = get_sa(df_cal[(~df_cal['fail']) & (df_cal['step'] == step)])
        assert X_cal_step.shape[0] > 0
        cal_scores = clf.score_samples(X_cal_step)
        #thresh[step] = cal_scores.max()
        thresh[step] = np.quantile(cal_scores, q)

    return thresh, total_time


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

def cd_rp(df_tr, df_cal, df_te, args):
    proj = GaussianRandomProjection(n_components=args.proj)

    tr_succ_proj_sa = proj.fit_transform(get_sa(df_tr[~df_tr['fail']]))
    tr_proj_sa = proj.transform(get_sa(df_tr))

    tr_succ_x = np.concatenate([df_tr[~df_tr['fail']]['t'].to_numpy()[:, np.newaxis], tr_succ_proj_sa], axis=1)
    tr_x = np.concatenate([df_tr['t'].to_numpy()[:, np.newaxis], tr_proj_sa], axis=1)
    p = CDPolynomial(tr_succ_x, degree=args.deg, verbose=True, basis='cheb', eps=0)
    df_tr['score'] = np.log(p(tr_x))

    def get_scores():
        te_proj_sa = proj.transform(get_sa(df_te))
        te_x = np.concatenate([df_te['t'].to_numpy()[:, np.newaxis], te_proj_sa], axis=1)
        return np.log(p(te_x))
    t = time_call(get_scores)
    df_te['score'] = get_scores()

    cal_proj_sa = proj.transform(get_sa(df_cal))
    cal_x = np.concatenate([df_cal['t'].to_numpy()[:, np.newaxis], cal_proj_sa], axis=1)
    df_cal['score'] = np.log(p(cal_x))
    #thresh = df_cal[~df_cal['fail']]['score'].max()
    thresh = np.quantile(df_cal[~df_cal['fail']]['score'], q)

    return np.full(n_steps, thresh), t

def log_reg(df_tr, df_cal, df_te, args):
    from sklearn.linear_model import LogisticRegression
    X_tr = get_tsa(df_tr)
    model = LogisticRegression().fit(X_tr, df_tr['fail'])
    df_te['score'] = model.predict_proba(get_tsa(df_te))[:,1]
    return np.full(n_steps, 0.5), 0.

def state_seq_recon(df_tr, df_cal, df_te, args):
    # load trained GRU model
    ckpt_dir = Path(f'./{args.env}/state_window/lightning_logs/{args.enc}/checkpoints')
    ae_file = glob.glob(str(ckpt_dir/'*.ckpt'))[-1]
    model = StateSeqAutoencoder.load_from_checkpoint(ae_file).to('cpu')
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


def state_seq_cd(df_tr, df_cal, df_te, args):
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
    return np.full(n_steps, thresh), t


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default='hopper')
    parser.add_argument("--method", type=str, default='cd')
    parser.add_argument("--enc")
    parser.add_argument("--proj", type=int)
    parser.add_argument("--deg", type=int, default=3)
    parser.add_argument("--plot", action='store_true')
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
    sa_cols = [f's{i}' for i in range(s_dim)] + [f'a{i}' for i in range(a_dim)]    
    dir = Path(f'./{args.env}')

    if args.enc is not None and args.method != 'state_seq_cd':
        ckpt_dir = dir/'lightning_logs'/args.enc/'checkpoints'
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

    def get_tsa(df):
        if ae is not None and args.method == 'cd_ae':
            z = ae.encode(torch.Tensor(df[sa_cols].to_numpy()))
            return np.concatenate([df['t'].to_numpy()[:,np.newaxis], 
                                   z.detach().numpy()], 
                                   axis=1)
        else:
            return df[['t'] + sa_cols].to_numpy()
    
    def get_sa(df):
        return df[sa_cols].to_numpy()

    #------- process data --------#
    df_tr = process_data(dir/'train.npz', s_dim, a_dim)
    df_te = process_data(dir/'test.npz', s_dim, a_dim)

    eps = df_tr['ep'].unique()
    n_cal_eps = max(1, int(0.1 * len(eps)))
    cal_eps = set(eps[:n_cal_eps])
    df_cal = df_tr[df_tr['ep'].isin(cal_eps)].copy()
    df_tr = df_tr[~df_tr['ep'].isin(cal_eps)].copy()

    #------- fit and predict -------#
    if args.method in ['cd', 'cd_ae']:
        thresh, t = cd_poly(df_tr, df_cal, df_te, args)
    elif args.method == 'dist':
        thresh, t = distance_based(df_tr, df_cal, df_te, args)
    elif args.method == 'ae_rec':
        thresh, t = ae_recon(df_tr, df_cal, df_te, args)
    elif args.method == 'cd_rp':
        thresh, t = cd_rp(df_tr, df_cal, df_te, args)
    elif args.method == 'log_reg':
        thresh, t = log_reg(df_tr, df_cal, df_te, args)
    elif args.method == 'state_seq_ae':
        thresh, t = state_seq_recon(df_tr, df_cal, df_te, args)
    elif args.method == 'state_seq_cd':
        thresh, t = state_seq_cd(df_tr, df_cal, df_te, args)


    df_te['thresh'] = df_te['step'].map(dict(enumerate(thresh)))
    df_te['pred'] = df_te['score'] >= df_te['thresh']


    #------- compute metrics --------#
    df_tr_succ = df_tr[~df_tr['fail']]
    df_tr_fail = df_tr[df_tr['fail']]
    df_te_succ = df_te[~df_te['fail']]
    df_te_fail = df_te[df_te['fail']]

    tp = df_te_fail.groupby('ep')['pred'].any() # true positives: failures that were flagged
    tn = ~df_te_succ.groupby('ep')['pred'].any() # true negatives: successes that weren't flagged

    tpr = tp.mean()
    tnr = tn.mean()
    bal_acc = (tpr + tnr) / 2
    prop_fail = df_te.groupby('ep')['fail'].first().mean()
    weight_acc = prop_fail * tpr + (1 - prop_fail) * tnr
    
    detected_eps = tp[tp].index
    first_pred_step = (
        df_te_fail.loc[df_te_fail['ep'].isin(detected_eps) & df_te_fail['pred']]
        .groupby('ep')['step']
        .min()
    )
    fail_step = (
        df_te_fail.loc[df_te_fail['ep'].isin(detected_eps)]
        .groupby('ep')['fail_step']
        .first()
    ).reindex(first_pred_step.index)

    assert (first_pred_step < n_steps).all()

    ttd = (fail_step - first_pred_step).mean()  # avg time to detect
    edr = np.mean(first_pred_step < fail_step)  # early detection rate


    #---------- Save results -------------#
    import json
    logfile = 'results.json'
    results = {
        'method': args.method,
        'env': args.env,
        'enc': args.enc,
        #'no_fail': no_fail,
        #'hid': hid_dim,
        #'lat': lat_dim,
        'proj': args.proj,
        'deg': args.deg,
        'tpr': tpr,
        'tnr': tnr,
        'bal_acc': bal_acc,
        'weight_acc': weight_acc,
        'ttd': ttd * skip / 10,  # out of 100
        'edr': edr,
        't': t
    }
    print(results)
    with open(logfile, 'a') as f:
        f.write(json.dumps(results) + '\n')

    if args.plot:
        make_plot(df_te, thresh)


