"""Throw-away: failure time vs initial-state value for each of hopper's 11 state dims."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist
import umap

d = np.load("data/hopper/fail_pred/data.npz")
fail_all = d["fail"]
init_all = d["X"][:, 0, :]
mask = fail_all < 1000
fail = fail_all[mask]
init = init_all[mask]
n_dims = init.shape[1]

df = pd.DataFrame(init, columns=[f"s{i}" for i in range(n_dims)])
df["fail"] = fail

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
for i, ax in enumerate(axes.flat):
    if i >= n_dims:
        ax.set_visible(False)
        continue
    col = f"s{i}"
    sns.regplot(
        data=df, x=col, y="fail", ax=ax,
        scatter_kws={"alpha": 0.2, "s": 8},
        line_kws={"color": "C1"},
        lowess=True,
    )
    ax.set_title(f"state dim {i}")
    ax.set_xlabel("initial value")
    ax.set_ylabel("failure time" if i % 4 == 0 else "")

fig.suptitle("Hopper: failure time vs initial state (per dim)")
fig.tight_layout()
out = "scripts/misc/hopper_fail_vs_init_state.png"
fig.savefig(out, dpi=120)
print(f"saved {out}")

init_std = StandardScaler().fit_transform(init)
emb = umap.UMAP(n_components=2, random_state=0).fit_transform(init_std)

fig2, ax2 = plt.subplots(figsize=(7, 6))
sc = ax2.scatter(emb[:, 0], emb[:, 1], c=fail, cmap="viridis", s=10, alpha=0.7)
fig2.colorbar(sc, ax=ax2, label="failure time")
ax2.set_xlabel("UMAP 1")
ax2.set_ylabel("UMAP 2")
ax2.set_title("Hopper init-state UMAP, colored by failure time")
fig2.tight_layout()
out2 = "scripts/misc/hopper_init_umap.png"
fig2.savefig(out2, dpi=120)
print(f"saved {out2}")

y = (fail_all < 1000).astype(int)
print(f"\nAUC: predicting fail<1000 from init state ({y.sum()}/{len(y)} positives)")
models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "rf": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
}
for name, m in models.items():
    aucs = cross_val_score(m, init_all, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  {name}: {aucs.mean():.3f} ± {aucs.std():.3f}")

rng = np.random.default_rng(0)


def binned_curve(d_pairs, delta_pairs, n_bins=25):
    """Mean of delta within log-spaced d bins. Returns (centers, mean, lo, hi)."""
    edges = np.geomspace(d_pairs.min() + 1e-12, d_pairs.max(), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    idx = np.digitize(d_pairs, edges) - 1
    mean = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 20:
            continue
        v = delta_pairs[sel]
        mean[b] = v.mean()
        lo[b] = np.quantile(v, 0.25)
        hi[b] = np.quantile(v, 0.75)
    return centers, mean, lo, hi


init_failed_std = StandardScaler().fit_transform(init)
d_pairs = pdist(init_failed_std)
fail_d_pairs = pdist(fail.reshape(-1, 1), metric="cityblock")
fail_shuf = rng.permutation(fail)
fail_d_shuf = pdist(fail_shuf.reshape(-1, 1), metric="cityblock")

init_all_std = StandardScaler().fit_transform(init_all)
d_pairs_all = pdist(init_all_std)
y_pairs = pdist(y.reshape(-1, 1), metric="cityblock")
y_shuf = rng.permutation(y)
y_pairs_shuf = pdist(y_shuf.reshape(-1, 1), metric="cityblock")

fig3, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

c, m, lo, hi = binned_curve(d_pairs, fail_d_pairs)
axL.fill_between(c, lo, hi, alpha=0.2, color="C0", label="real IQR")
axL.plot(c, m, color="C0", lw=2, label="real mean")
c, m, lo, hi = binned_curve(d_pairs, fail_d_shuf)
axL.plot(c, m, color="C3", ls="--", lw=2, label="shuffled mean")
axL.set_xscale("log")
axL.set_yscale("log")
axL.set_xlabel("||Δ init-state||  (standardized)")
axL.set_ylabel("|Δ fail|  (steps)")
axL.set_title(f"Continuous: failures only (n={len(fail)})")
axL.legend()

c, m, lo, hi = binned_curve(d_pairs_all, y_pairs)
axR.plot(c, m, color="C0", lw=2, label="real")
c, m, lo, hi = binned_curve(d_pairs_all, y_pairs_shuf)
axR.plot(c, m, color="C3", ls="--", lw=2, label="shuffled")
axR.set_xscale("log")
axR.set_xlabel("||Δ init-state||  (standardized)")
axR.set_ylabel("P(label flip)")
axR.set_title(f"Binary: fail<1000 (n={len(y)})")
axR.legend()

fig3.suptitle("Local smoothness of init-state → fail map")
fig3.tight_layout()
out3 = "scripts/misc/hopper_local_smoothness.png"
fig3.savefig(out3, dpi=120)
print(f"saved {out3}")
