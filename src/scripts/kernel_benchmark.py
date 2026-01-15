from utils.signals import *
from algs.kern_cd import KernCD
from algs.kernels import *
from experiments.safety_monitor import *
from algs.dim_red import *
#from algs.cd_poly import CDPolynomial   

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve


CAL = 0.3  # calibration set size
N_PERIOD = 1
N_TRIALS = 5


def estimate_window_length(x_tr: np.ndarray) -> int:
    freq = np.array([[estimate_freq(x_tr[i,:,j], fs=1) for i in range(len(x_tr))] for j in range(x_tr.shape[-1])]).T

    #sns.violinplot(freq)
    #plt.show()

    fhat = np.mean(freq)
    window = int(np.ceil(N_PERIOD / fhat))
    print(f'freq.: {fhat}, # periods: {N_PERIOD}, window: {window}')
    return window


def precision_at_recall(y_true, y_score, recall_level=0.95):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    mask = recall >= recall_level
    if not np.any(mask):  
        return 0.0, None  # cannot reach required recall
    
    idx = np.argmax(precision[mask])
    precision_vals = precision[mask]
    threshold_vals = thresholds[mask[:-1]]  # thresholds is len-1
    return precision_vals[idx], threshold_vals[idx]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--alg', type=str)
    args = parser.parse_args()

    sm_cfg = SM_CFG[args.env]

    task = SafetyMonitor(args.env, sm_cfg)
    x_tr, x_te = task.get_train_test()

    from utils.signals import low_pass
    alpha = 0.8
    x_tr = low_pass(x_tr, alpha)
    x_te = low_pass(x_te, alpha)

    # estimate data-dependent window length
    window = estimate_window_length(x_tr)
    if args.alg == 'fft':
        window = window
    if args.alg == 'sig':
        window = window // 2
    if args.alg == 'pca':
        window = window
    # truncate x_te
    assert window <= sm_cfg.win
    x_te = x_te[:, -window:]

    scores = []
    for i in range(N_TRIALS):
        # ?????
        starts = np.random.randint(low=100, high=x_tr.shape[1] - window + 1, size=len(x_tr))
        idx = starts[:, None] + np.arange(window)[None, :]
        x_tr_trial = x_tr[np.arange(len(x_tr))[:, None], idx]
        x_tr_trial, x_cal = train_test_split(x_tr_trial, test_size=CAL)
        perm = np.random.permutation(len(x_tr_trial))
        x_tr_trial = x_tr_trial[:100]  # ?
        x_te_trial = x_te
        print(x_tr_trial.shape, x_cal.shape, x_te.shape)
        #print(x_tr_trial.shape, x_te.shape)

        if args.alg == 'fft':
            model = KernCD(GaussFFT(gamma=.5), lam=1e-3).fit(x_tr_trial)
        if args.alg == 'sig':
            model = KernCD(SigKernel(gamma=0.001), lam=1e-3).fit(x_tr_trial)
        if args.alg == 'pca':
            pca = PCA_FFT(k1=0.9, k2=3)
            x_tr_trial = pca.fit_transform(x_tr_trial)
            x_cal = pca.transform(x_cal)
            x_te_trial = pca.transform(x_te_trial)
            print(f'pca-fft output: {x_tr_trial.shape}')
            model = CDPolynomial(x_tr_trial, degree=5, verbose=True)
        if args.alg == 'kld':

            pca = PCA(0.95)
            x_tr_trial = pca.fit_transform(x_tr_trial.reshape((len(x_tr_trial), -1)))
            x_cal = pca.transform(x_cal.reshape((len(x_cal), -1)))
            x_te_trial = pca.transform(x_te_trial.reshape(len(x_te_trial), -1))

            print(x_tr_trial.shape)

            model = KernCD(RBF(gamma=0.005), lam=1e-5).fit(x_tr_trial)
            

        q = np.quantile(model.predict(x_cal), q=.95)
        print(f'Finished training')
        #y_pred = model.predict(x_te_trial) > q
        y_pred = model.predict(x_te_trial) > q

        #y_score = model.predict(x_te_trial)
        y_true = task.get_test_labels()
        #score, q = fpr_at_recall(y_true, y_score)
        #score, thresh = precision_at_recall(y_true, y_score, recall_level=0.9)
        score = confusion_matrix(y_true, y_pred, normalize='true')
        print(f'Trial {i}: score {score}')
        scores.append(score)

    #print(f'Score: {np.mean(scores):.3f} ± {np.std(scores)/np.sqrt(len(scores))}')
    scores = np.array(scores)
    scores_mean = np.mean(scores, axis=0).ravel()
    scores_std = np.std(scores, axis=0).ravel() / np.sqrt(N_TRIALS)
    print(f'TN: {scores_mean[0] * 100:.2f} ± {scores_std[0] * 100:.2f}')
    print(f'FP: {scores_mean[1] * 100:.2f} ± {scores_std[1] * 100:.2f}')
    print(f'FN: {scores_mean[2] * 100:.2f} ± {scores_std[2] * 100:.2f}')
    print(f'TP: {scores_mean[3] * 100:.2f} ± {scores_std[3] * 100:.2f}')

    output = "".join([
        f" & {scores_mean[i] * 100:.2f} $\\pm$ {scores_std[i] * 100:.2f}" for i in range(4)
    ]) + "\\\\"
    print(output)