import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser
from pathlib import Path

import glob
from sklearn.metrics import precision_recall_curve, auc
from algs.cd_poly import CDPolynomial
from algs.pair_ae import PairAE
import lightning as L
from torch.utils.data import DataLoader
from sklearn.random_projection import GaussianRandomProjection
from algs.pair_ae import PairAE

n_eps = 1000
n_steps = 50
skip = 20

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
    return df


def make_plot(df_te, thresh):
    for i in df_te[~df_te['fail']]['ep'].unique():
        plt.plot(range(n_steps), df_te[~df_te['fail']][df_te[~df_te['fail']]['ep'] == i]['score'], color='green', alpha=0.2)
    for i in df_te[df_te['fail']]['ep'].unique():
        plt.plot(range(n_steps), df_te[df_te['fail']][df_te[df_te['fail']]['ep'] == i]['score'], color='red', alpha=0.2)
    plt.plot(thresh, color='blue')
    plt.show()

#---------- Methods to benchmark --------------------#

def cd_poly(df_tr, df_te, args):
    p = CDPolynomial(get_tsa(df_tr[~df_tr['fail']]), degree=args.deg, verbose=True, basis='cheb', eps=0)
    df_tr['score'] = np.log(p(get_tsa(df_tr)))
    df_te['score'] = np.log(p(get_tsa(df_te)))

    thresh = df_tr[~df_tr['fail']]['score'].max()
    return np.full(n_steps, thresh)


def distance_based(df_tr, df_te, args):
    from algs.distance_baseline import DistanceBasedOneClass

    clfs = []
    for step in range(n_steps):
        clf = DistanceBasedOneClass()
        clf.fit(get_sa(df_tr[(~df_tr['fail']) & (df_tr['step'] == step)]))
        clfs.append(clf)
        df_te.loc[df_te['step'] == step, 'score'] = clf.score_samples(get_sa(df_te[df_te['step'] == step]))

    thresh = np.array([clfs[i].threshold for i in range(n_steps)])
    return thresh

def ae_recon(df_tr, df_te, args):
    assert ae.hparams.no_fail
    x_tr = torch.Tensor(get_sa(df_tr).astype(np.float32))
    x_te = torch.Tensor(get_sa(df_te).astype(np.float32))
    with torch.no_grad():
        x_tr_hat, _ = ae(x_tr)
        x_te_hat, _ = ae(x_te)
    df_tr['score'] = ((x_tr_hat - x_tr) ** 2).mean(dim=1).detach().numpy()
    df_te['score'] = ((x_te_hat - x_te) ** 2).mean(dim=1).detach().numpy()
    thresh = df_tr[~df_tr['fail']]['score'].max()
    return np.full(n_steps, thresh)

def cd_rp(df_tr, df_te, args):
    proj = GaussianRandomProjection(n_components=args.proj)

    tr_succ_proj_sa = proj.fit_transform(get_sa(df_tr[~df_tr['fail']]))
    tr_proj_sa = proj.transform(get_sa(df_tr))
    te_proj_sa = proj.transform(get_sa(df_te))

    tr_succ_x = np.concatenate([df_tr[~df_tr['fail']]['t'].to_numpy()[:, np.newaxis], tr_succ_proj_sa], axis=1)
    tr_x = np.concatenate([df_tr['t'].to_numpy()[:, np.newaxis], tr_proj_sa], axis=1)
    te_x = np.concatenate([df_te['t'].to_numpy()[:, np.newaxis], te_proj_sa], axis=1)

    p = CDPolynomial(tr_succ_x, degree=args.deg, verbose=True, basis='cheb', eps=0)
    df_tr['score'] = np.log(p(tr_x))
    df_te['score'] = np.log(p(te_x))

    thresh = df_tr[~df_tr['fail']]['score'].max()
    return np.full(n_steps, thresh)


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
    dir = Path(f'./{args.env}')

    if args.enc is not None:
        ckpt_dir = dir/'lightning_logs'/args.enc/'checkpoints'
        ae_file = glob.glob(str(ckpt_dir/'*.ckpt'))[-1]
        ae = PairAE.load_from_checkpoint(ae_file).to('cpu')
        no_fail = ae.hparams.no_fail
    else:
        ae = None
        no_fail = None

    def get_tsa(df):
        sa_cols = [f's{i}' for i in range(s_dim)] + [f'a{i}' for i in range(a_dim)]    
        if ae is not None and args.method == 'cd_ae':
            z = ae.encode(torch.Tensor(df[sa_cols].to_numpy()))
            return np.concatenate([df['t'].to_numpy()[:,np.newaxis], 
                                   z.detach().numpy()], 
                                   axis=1)
        else:
            return df[['t'] + sa_cols].to_numpy()
    
    def get_sa(df):
        return df[[f's{i}' for i in range(s_dim)] + [f'a{i}' for i in range(a_dim)]].to_numpy()

    #------- process data --------#
    df_tr = process_data(dir/'train.npz', s_dim, a_dim)
    df_te = process_data(dir/'test.npz', s_dim, a_dim)

    #------- fit and predict -------#
    if args.method in ['cd', 'cd_ae']:
        thresh = cd_poly(df_tr, df_te, args)
    elif args.method == 'dist':
        thresh = distance_based(df_tr, df_te, args)
    elif args.method == 'ae_rec':
        thresh = ae_recon(df_tr, df_te, args)
    elif args.method == 'cd_rp':
        thresh = cd_rp(df_tr, df_te, args)

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
    
    detected_eps = detected_eps = tp[tp].index
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
        'no_fail': no_fail,
        'deg': args.deg,
        'tpr': tpr,
        'tnr': tnr,
        'bal_acc': bal_acc,
        'weight_acc': weight_acc,
        'ttd': ttd,
        'edr': edr
    }
    print(results)
    with open(logfile, 'a') as f:
        f.write(json.dumps(results) + '\n')

    if args.plot:
        make_plot(df_te, thresh)


