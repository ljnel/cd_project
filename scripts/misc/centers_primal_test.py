"""
Fixed-center primal feature map: residual, primal Mahalanobis, and the
full (inducing-point) GP posterior variance = Mahalanobis + residual.

  psi(x) = K_pp^{-1/2} k_p(x)            (data-independent, fixed before data)
  Delta(x) = K(x,x) - ||psi(x)||^2        residual = ||P_S^perp Phi(x)||^2
  maha(x)  = lam <psi, (Sigma_p+lam I)^{-1} psi>   in-subspace GP variance
  GPvar(x) = maha(x) + Delta(x)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

rng = np.random.default_rng(0)

# ---- RBF kernel, normalized so K(x,x)=1 ----
gamma = 0.5
def kernel(A, B):
    sq = (np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T)
    return np.exp(-gamma * np.maximum(sq, 0.0))

# ---- box and centers (chosen BEFORE any data) ----
lo, hi = -4.0, 4.0
p = 30

def make_centers(scheme="fps", p=p, n_cand=4000):
    """'uniform' = i.i.d. uniform; 'fps' = farthest-point (min fill distance)."""
    if scheme == "uniform":
        return rng.uniform(lo, hi, size=(p, 2))
    # farthest-point sampling over a dense uniform candidate pool
    cand = rng.uniform(lo, hi, size=(n_cand, 2))
    chosen = [int(rng.integers(n_cand))]
    d2 = np.sum((cand - cand[chosen[0]])**2, axis=1)   # dist^2 to nearest center
    for _ in range(p - 1):
        i = int(np.argmax(d2))                         # farthest from all chosen
        chosen.append(i)
        d2 = np.minimum(d2, np.sum((cand - cand[i])**2, axis=1))
    return cand[chosen]

centers = make_centers("fps")

# ---- fixed primal feature map psi(x) = K_pp^{-1/2} k_p(x) ----
Kpp = kernel(centers, centers)
evals, evecs = np.linalg.eigh(Kpp)
keep = evals > 1e-10 * evals.max()
Bmat = evecs[:, keep] / np.sqrt(evals[keep])          # (p, r)

def psi(X):
    return kernel(X, centers) @ Bmat                  # (n, r)

def residual(X):
    P = psi(X)
    return np.maximum(1.0 - np.sum(P**2, 1), 0.0)      # 1 - ||psi||^2

# ---- evaluation grid ----
g = np.linspace(lo, hi, 220)
GX, GY = np.meshgrid(g, g)
grid = np.c_[GX.ravel(), GY.ravel()]

# ---- multimodal data via make_blobs, inside the box ----
Xd, _ = make_blobs(n_samples=700,
                   centers=[[-2.3, -2.0], [2.2, 2.2], [-2.0, 2.3]],
                   cluster_std=0.45, random_state=0)
n = len(Xd)

# ---- (uncentered) feature-space covariance = GP posterior-variance form ----
lam = 1e-2
Pd = psi(Xd)
Sig = (Pd.T @ Pd) / n                                  # (r, r)
r = Sig.shape[0]
M = lam * np.linalg.inv(Sig + lam * np.eye(r))         # lam (Sigma+lam I)^{-1}

def maha(X):                                           # in-subspace GP variance
    P = psi(X)
    return np.einsum("ij,jk,ik->i", P, M, P)

# ---- evaluate fields on grid ----
D_grid    = residual(grid).reshape(GX.shape)
maha_grid = maha(grid).reshape(GX.shape)
gp_grid   = maha_grid + D_grid

# ---- plot ----
fig, ax = plt.subplots(1, 3, figsize=(16, 5.3))
box_kw = dict(fill=False, edgecolor="white", lw=1.5, ls="--")

im0 = ax[0].pcolormesh(GX, GY, D_grid, cmap="magma", shading="auto")
ax[0].scatter(centers[:, 0], centers[:, 1], s=28, c="cyan",
              edgecolor="k", linewidths=0.4, label="centers")
ax[0].add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, **box_kw))
ax[0].set_title(r"1. Residual  $\Delta(x)=K(x,x)-\|\psi(x)\|^2$"
                "\n(farthest-point centers)")
ax[0].legend(loc="upper right", fontsize=8)
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].pcolormesh(GX, GY, maha_grid, cmap="viridis", shading="auto")
ax[1].scatter(Xd[:, 0], Xd[:, 1], s=4, c="white", alpha=0.45, linewidths=0)
ax[1].add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, **box_kw))
ax[1].set_title(r"2. Primal Mahalanobis  $\lambda\,\psi^\top(\Sigma_p+\lambda I)^{-1}\psi$"
                "\n(no residual term)")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

im2 = ax[2].pcolormesh(GX, GY, gp_grid, cmap="viridis", shading="auto")
ax[2].scatter(Xd[:, 0], Xd[:, 1], s=4, c="white", alpha=0.45, linewidths=0)
ax[2].add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, **box_kw))
ax[2].set_title("3. Full GP posterior variance\nMahalanobis + residual")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("fixed_center_primal.png", dpi=130)
print("p centers =", p, " feature dim r =", r, " n_data =", n)
print("residual  range:", D_grid.min().round(3), D_grid.max().round(3))
print("maha      range:", maha_grid.min().round(3), maha_grid.max().round(3))
print("gp var    range:", gp_grid.min().round(3), gp_grid.max().round(3))

# fill distance (radius of largest empty gap) over the grid: lower = more even
def fill_distance(C):
    d2 = np.min(((grid[:, None, :] - C[None, :, :])**2).sum(-1), axis=1)
    return np.sqrt(d2.max())
print("\nfill distance (max gap radius), lower is more even:")
print("   uniform centers:", round(fill_distance(make_centers('uniform')), 3))
print("   fps     centers:", round(fill_distance(centers), 3))