"""How predictable is the CD pointwise-q90 indicator label 5 steps ahead,
given a window of k past obs?

- Fit CD (deg 7, cheb) on (r, z) of TRAINING surviving trajectories only.
- q90 = 90% quantile of CD on training (r, z) points only.
- Binary label at step t: y_t = 1[CD(r_t, z_t) <= q90].
- For each k in K_LIST: input = obs[t-k+1:t+1] flattened (27*k feats),
  target = y_{t+5}, filtered to y_t == 1 (start inside).
- Train MLP on train pairs, eval on test pairs (split by trajectory).
- Report test AUC / accuracy vs k. Logistic-regression baseline uses
  current (r, z) only.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

from data.io import load
from data.dataset import survived
from algs.cd_poly import CDPolynomial
from utils.paths import get_output_dir

# Config
T_START = 50
DEGREE = 7
HORIZON = 5
K_LIST = [1, 5, 10]
# Scale hidden width with history length so capacity/input-dim stays comparable.
# 2 hidden layers, each of width WIDTH_PER_STEP * k.
WIDTH_PER_STEP = 64
TRAIN_FRAC = 0.7
SEED = 0
STEP_STRIDE = 5  # subsample (input, target) pairs in time for speed

# 1. Load and split surviving trajectories.
ds = survived(load(env='ant', name='base', obs_only=True))
T_full = ds.X.shape[1]
X_all = ds.X[:, T_START:]  # (N, T, 27)
N, T, D = X_all.shape
print(f"surviving trajectories: {N}, steps per traj: {T}, obs dim: {D}")

rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
n_train = int(TRAIN_FRAC * N)
tr_idx, te_idx = perm[:n_train], perm[n_train:]
X_tr_obs, X_te_obs = X_all[tr_idx], X_all[te_idx]
print(f"train trajs: {len(tr_idx)}, test trajs: {len(te_idx)}")

# 2. Extract (r, z) per step.
def rz(obs):
    z = obs[..., 0]
    r = np.sqrt(obs[..., 2] ** 2 + obs[..., 3] ** 2)
    return r, z

r_tr, z_tr = rz(X_tr_obs)
r_te, z_te = rz(X_te_obs)

# 3. Fit CD on TRAIN (r, z) only.
train_rz = np.stack([r_tr.ravel(), z_tr.ravel()], axis=-1)
cd = CDPolynomial(train_rz, degree=DEGREE, basis='cheb', method='qr')
cd_train_vals = np.asarray(cd(train_rz))
# Threshold = 90th quantile of (per-trajectory max of CD on training trajs).
cd_train_per_traj_max = cd_train_vals.reshape(len(tr_idx), T).max(axis=1)
thr = float(np.quantile(cd_train_per_traj_max, 0.9))
print(f"CD: deg={DEGREE}, mean={cd.mean:.3g}, q90(per-traj max)={thr:.3g}")

# 4. Per-step labels for train + test.
def labels(r, z):
    pts = np.stack([r.ravel(), z.ravel()], axis=-1)
    vals = np.asarray(cd(pts)).reshape(r.shape)
    return (vals <= thr).astype(np.int8)

y_tr = labels(r_tr, z_tr)  # (n_train, T)
y_te = labels(r_te, z_te)  # (n_test, T)
print(f"label balance — train inside: {y_tr.mean():.3f}, test inside: {y_te.mean():.3f}")

# 5. For each k, build (input, target) pairs.
def build_pairs(X_obs, y, k, h):
    """Window of k past obs at step t -> label at step t+h.

    Strict filter: ALL k window labels y[t-k+1 .. t] must be 1 (fully inside).
    Returns (N_pairs, k*D) inputs and (N_pairs,) targets, subsampled by STEP_STRIDE.
    """
    _, T_, D_ = X_obs.shape
    inputs, targets = [], []
    valid_t = np.arange(k - 1, T_ - h, STEP_STRIDE)
    for t in valid_t:
        mask = y[:, t - k + 1 : t + 1].all(axis=1)  # full window inside
        if mask.sum() == 0:
            continue
        win = X_obs[mask, t - k + 1 : t + 1]  # (m, k, D)
        inputs.append(win.reshape(mask.sum(), k * D_))
        targets.append(y[mask, t + h])
    return np.concatenate(inputs), np.concatenate(targets).astype(int)

results = []
for k in K_LIST:
    Xtr, ytr = build_pairs(X_tr_obs, y_tr, k, HORIZON)
    Xte, yte = build_pairs(X_te_obs, y_te, k, HORIZON)
    sc = StandardScaler().fit(Xtr)
    width = WIDTH_PER_STEP * k
    mlp = MLPClassifier(hidden_layer_sizes=(width, width), max_iter=200,
                        early_stopping=True, random_state=SEED, batch_size=256)
    mlp.fit(sc.transform(Xtr), ytr)
    p = mlp.predict_proba(sc.transform(Xte))[:, 1]
    auc = roc_auc_score(yte, p)
    acc = accuracy_score(yte, (p > 0.5).astype(int))
    maj = max(yte.mean(), 1 - yte.mean())
    results.append((k, width, len(ytr), len(yte), yte.mean(), auc, acc, maj))
    print(f"  k={k:>2}  width={width:>4}  n_tr={len(ytr):>6}  n_te={len(yte):>6}  "
          f"pos={yte.mean():.3f}  AUC={auc:.4f}  acc={acc:.4f}  maj={maj:.4f}")

# 6. Logistic regression baseline using current (r, z) only.
def rz_only(X_obs, y, h):
    n, T_, _ = X_obs.shape
    valid_t = np.arange(0, T_ - h, STEP_STRIDE)
    Xs, ys = [], []
    for t in valid_t:
        mask = y[:, t] == 1
        if mask.sum() == 0:
            continue
        z = X_obs[mask, t, 0]
        r = np.sqrt(X_obs[mask, t, 2] ** 2 + X_obs[mask, t, 3] ** 2)
        Xs.append(np.stack([r, z], axis=-1))
        ys.append(y[mask, t + h])
    return np.concatenate(Xs), np.concatenate(ys).astype(int)

Xtr_b, ytr_b = rz_only(X_tr_obs, y_tr, HORIZON)
Xte_b, yte_b = rz_only(X_te_obs, y_te, HORIZON)
sc_b = StandardScaler().fit(Xtr_b)
lr = LogisticRegression(max_iter=500).fit(sc_b.transform(Xtr_b), ytr_b)
p_b = lr.predict_proba(sc_b.transform(Xte_b))[:, 1]
auc_b = roc_auc_score(yte_b, p_b)
print(f"baseline (logreg on current (r,z)): AUC={auc_b:.4f}")

# 7. Plot AUC vs k.
ks = [r[0] for r in results]
widths = [r[1] for r in results]
aucs = [r[5] for r in results]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ks, aucs, 'o-', lw=2, label=f'MLP, hidden = ({WIDTH_PER_STEP}·k, {WIDTH_PER_STEP}·k)')
for x, y, w in zip(ks, aucs, widths):
    ax.annotate(f'w={w}', (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8)
ax.axhline(auc_b, color='C3', ls='--', label='logreg on current $(r, z)$ only')
ax.set_xlabel('history length $k$ (steps of past obs)')
ax.set_ylabel(f'test AUC at horizon h={HORIZON}')
ax.set_title('Predicting CD per-traj-max-$q_{90}$ label 5 steps ahead from inside')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
out = get_output_dir() / "ant_label_predictability.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
