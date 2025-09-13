import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser

import glob
from sklearn.metrics import precision_recall_curve, auc
from algs.cd_poly import CDPolynomial
from algs.pair_ae import PairAE
import lightning as L
from torch.utils.data import DataLoader
from sklearn.random_projection import GaussianRandomProjection

n_eps = 1000
n_steps = 50
skip = 20


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


## Each method gets df_tr and df_te; it has to set scores and return a threshold

def raw_cd_poly(df_tr, df_te,
                deg):
    p = CDPolynomial(get_tsa(df_tr[~df_tr['fail']]), degree=deg, verbose=True, basis='cheb', eps=0)
    df_tr['score'] = np.log(p(get_tsa(df_tr)))
    df_te['score'] = np.log(p(get_tsa(df_te)))

    thresh = df_tr[~df_tr['fail']]['score'].max()
    return np.full(n_steps, thresh)


def distance_based(df_tr, df_te):
    from algs.distance_baseline import DistanceBasedOneClass

    clfs = []
    for step in range(n_steps):
        clf = DistanceBasedOneClass()
        clf.fit(get_tsa(df_tr[(~df_tr['fail']) & (df_tr['step'] == step)]))
        clfs.append(clf)
        df_te.loc[df_te['step'] == step, 'score'] = clf.score_samples(get_tsa(df_te[df_te['step'] == step]))

    thresh = np.array([clfs[i].threshold for i in range(n_steps)])
    return thresh


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default='hopper')
    parser.add_argument("--method", type=str, default='cd')
    parser.add_argument("--lat", type=int, default=10)
    parser.add_argument("--deg", type=int, default=3)
    args = parser.parse_args()

    #----- get the env -----#
    if args.env == 'cartpole':
        s_dim, a_dim = 4, 1

    elif args.env == 'hopper':
        s_dim, a_dim = 11, 3


    def get_tsa(df):
        return df[['t'] + [f's{i}' for i in range(s_dim)] + [f'a{i}' for i in range(a_dim)]].to_numpy()

    #------- process data --------#
    df_tr = process_data('train.npz', s_dim, a_dim)
    df_te = process_data('test.npz', s_dim, a_dim)

    #------- fit and predict -------#
    if args.method == 'cd':
        thresh = raw_cd_poly(df_tr, df_te, args.deg)
    elif args.method == 'dist':
        thresh = distance_based(df_tr, df_te)

    df_te['thresh'] = df_te['step'].map(dict(enumerate(thresh)))
    df_te['pred'] = df_te['score'] >= df_te['thresh']


    #------- compute metrics --------#
    df_tr_succ = df_tr[~df_tr['fail']]
    df_tr_fail = df_tr[df_tr['fail']]
    df_te_succ = df_te[~df_te['fail']]
    df_te_fail = df_te[df_te['fail']]

    tp = ~df_te_succ.groupby('ep')['pred'].any() # true positives
    tn = df_te_fail.groupby('ep')['pred'].any()

    tpr = tp.mean()
    tnr = tn.mean()
    bal_acc = (tpr + tnr) / 2
    prop_succ = df_te.groupby('ep')['fail'].first().mean()
    weight_acc = prop_succ * tpr + (1 - prop_succ) * tnr
    
    df_te_fail_step = df_te_fail[['ep', 'fail_step']].drop_duplicates()['fail_step'].to_numpy()
    df_te_fail_pred = df_te_fail.loc[df_te_fail.groupby('ep')['pred'].idxmax(), 'step']

    print(df_te_fail_pred.max())

    assert (df_te_fail_pred < n_steps).all()
    #prop_caught = np.mean(df_te_fail_pred < df_te_fail_step)
    ttd = (df_te_fail_step - df_te_fail_pred).mean()



    ### Save results ###
    import json
    logfile = 'results.json'
    results = {
        'method': args.method,
        'env': args.env,
        'lat': args.lat,
        'deg': args.deg,
        'tpr': tpr,
        'tnr': tnr,
        'bal_acc': bal_acc,
        'weight_acc': weight_acc,
        'ttd': ttd
    }
    with open(logfile, 'a') as f:
        f.write(json.dumps(results) + '\n')


