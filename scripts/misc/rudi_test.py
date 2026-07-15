r"""
Regularized spectral support estimator (Rudi, De Vito, Verri, Odone, 2017)
with a greedy pivoted-Cholesky (Nystrom) low-rank approximation.

The decision function is the squared distance from phi(x) to the regularized
data subspace, computed exactly inside the pivot subspace S and via the exact
residual outside it:

    F(x)^2 = || (I - r(Sigma_p)) psi(x) ||^2  +  ( K(x,x) - ||psi(x)||^2 )
             \________ in-subspace, A(x) _______/   \____ residual, Delta(x) ___/

Spectral form (Sigma_p = W diag(sigma_j) W^T, a = W^T psi(x)):

    F(x)^2 = K(x,x) - sum_j [ 2 r(sigma_j) - r(sigma_j)^2 ] a_j(x)^2

A point is "in the support" when F(x)^2 <= tau. Larger F^2 => more anomalous.
Hard-cutoff filter recovers Nystrom-KPCA reconstruction error; Tikhonov is the
soft default.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


# --------------------------------------------------------------------------- #
# kernels (stationary, normalized so K(x,x) = 1)
# --------------------------------------------------------------------------- #
def _kernel(X, Y, kind, gamma):
    if kind == "rbf":                       # Gaussian (NOT separating)
        return np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    if kind == "abel":                      # exp(-gamma |x-w|_2)  -- separating
        return np.exp(-gamma * cdist(X, Y, "euclidean"))
    if kind == "laplacian":                 # exp(-gamma |x-w|_1)  -- separating
        return np.exp(-gamma * cdist(X, Y, "cityblock"))
    raise ValueError(f"unknown kernel {kind!r}")


def _kernel_diag(X, kind):
    # all three kernels above are normalized: K(x, x) = 1
    return np.ones(X.shape[0])


# --------------------------------------------------------------------------- #
# greedy pivoted Cholesky -> pivot indices
# --------------------------------------------------------------------------- #
def _greedy_pivots(X, kind, gamma, n_pivots, tol):
    """Deterministic greedy pivoting: repeatedly take the point of largest
    current residual diagonal (Schur complement diagonal == Delta(X_i))."""
    n = X.shape[0]
    d = _kernel_diag(X, kind).astype(float)        # residual diagonal
    L = np.zeros((n, n_pivots))                     # partial Cholesky factor
    pivots = []
    for j in range(n_pivots):
        i = int(np.argmax(d))
        if d[i] <= tol:                             # subspace already captures data
            break
        pivots.append(i)
        col = _kernel(X, X[i:i + 1], kind, gamma).ravel()   # K(X, X_i)
        L[:, j] = (col - L[:, :j] @ L[i, :j]) / np.sqrt(d[i])
        d = np.maximum(d - L[:, j] ** 2, 0.0)
    return np.asarray(pivots, dtype=int)


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
class RudiSupportEstimator:
    """
    Parameters
    ----------
    kernel : {'abel', 'laplacian', 'rbf'}
        'abel'/'laplacian' satisfy the separating condition (consistency holds);
        'rbf' (Gaussian) does NOT separate full-dimensional supports.
    gamma : float
        Kernel bandwidth parameter.
    filter : {'tikhonov', 'hardcut'}
        'tikhonov' r(s)=s/(s+reg) is the soft default; 'hardcut' keeps the top
        `n_components` directions and recovers KPCA reconstruction error.
    reg : float
        Tikhonov regularization lambda (ignored for 'hardcut').
    n_components : int | None
        Number of kept directions for 'hardcut'.
    n_pivots : int
        Number p of greedy pivots (low-rank size).
    center : bool
        Center in feature space (standard for support estimation).
    threshold_quantile : float in (0, 1]
        tau is this quantile of training scores; 1.0 == paper's max rule.
    rcond : float
        Relative eigenvalue floor for K_pp^{-1/2}.
    """

    def __init__(self, kernel="abel", gamma=1.0, filter="tikhonov", reg=1e-3,
                 n_components=None, n_pivots=100, center=True,
                 threshold_quantile=1.0, rcond=1e-10):
        self.kernel = kernel
        self.gamma = gamma
        self.filter = filter
        self.reg = reg
        self.n_components = n_components
        self.n_pivots = n_pivots
        self.center = center
        self.threshold_quantile = threshold_quantile
        self.rcond = rcond

    # -- filter weights w_j = 1 - (1 - r(sigma_j))^2 = 2 r - r^2 ------------- #
    def _weights(self, sigma):
        if self.filter == "tikhonov":
            r = sigma / (sigma + self.reg)
        elif self.filter == "hardcut":
            m = self.n_components or len(sigma)
            order = np.argsort(sigma)[::-1]
            r = np.zeros_like(sigma)
            r[order[:m]] = 1.0
        else:
            raise ValueError(f"unknown filter {self.filter!r}")
        return 2.0 * r - r ** 2, r          # weight_j, r_j

    # -- Nystrom feature map psi(x) = K_pp^{-1/2} k_p(x) -------------------- #
    def _psi(self, X):
        kpx = _kernel(self.pivots_X_, X, self.kernel, self.gamma)   # (p, n)
        return (self.B_.T @ kpx).T                                  # (n, r)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        p = min(self.n_pivots, n)

        # 1. greedy pivots and their feature matrix
        piv = _greedy_pivots(X, self.kernel, self.gamma, p, self.rcond)
        self.pivots_X_ = X[piv]
        Kpp = _kernel(self.pivots_X_, self.pivots_X_, self.kernel, self.gamma)

        # 2. B = K_pp^{-1/2} on the numerical range  (psi has dim r <= p)
        evals, evecs = np.linalg.eigh(Kpp)
        keep = evals > self.rcond * evals.max()
        self.B_ = evecs[:, keep] / np.sqrt(evals[keep])            # (p, r)

        # 3. coordinates of the data, optional centering
        Psi = self._psi(X)                                          # (n, r)
        self.mean_ = Psi.mean(0) if self.center else np.zeros(Psi.shape[1])
        Psi_c = Psi - self.mean_

        # 4. covariance in coordinates and its spectrum
        Sigma = (Psi_c.T @ Psi_c) / n                               # (r, r)
        self.sigma_, self.W_ = np.linalg.eigh(Sigma)                # ascending
        self.weights_, _ = self._weights(self.sigma_)

        # 5. data-driven threshold tau
        train_scores = self._scores(X)
        q = self.threshold_quantile
        self.threshold_ = (train_scores.max() if q >= 1.0
                           else np.quantile(train_scores, q))
        return self

    # -- raw squared-distance score F(x)^2 (larger => more anomalous) ------- #
    def _scores(self, X):
        X = np.asarray(X, dtype=float)
        psi = self._psi(X)                                          # (n, r)
        Kxx = _kernel_diag(X, self.kernel)                          # (n,)
        a = (psi - self.mean_) @ self.W_                            # (n, r)

        # in-subspace term uses (1 - r)^2 = 1 - weight; residual uses
        # uncentered ||psi||^2 (exact, assumes feature-mean lies in span S).
        one_minus_r_sq = 1.0 - self.weights_                        # = (1 - r)^2
        A = (a ** 2) @ one_minus_r_sq                               # in-subspace
        Delta = Kxx - np.einsum("ij,ij->i", psi, psi)               # residual >=0
        return A + np.maximum(Delta, 0.0)

    # -- sklearn-style API -------------------------------------------------- #
    def score_samples(self, X):
        """Higher = more normal (sklearn convention): returns -F(x)^2."""
        return -self._scores(X)

    def decision_function(self, X):
        """Signed: > 0 inside the estimated support, < 0 outside."""
        return self.threshold_ - self._scores(X)

    def predict(self, X):
        """+1 inlier (in support), -1 outlier."""
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X):
        """Convenience: raw anomaly distances F(x)^2."""
        return self._scores(X)


# --------------------------------------------------------------------------- #
# self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
 
    # non-uniform circle: dense right arc, sparse left arc; support = full circle
    def sample_circle(n):
        theta = np.where(rng.random(n) < 0.85,
                         rng.uniform(-np.pi / 2, np.pi / 2, n),     # dense right
                         rng.uniform(np.pi / 2, 3 * np.pi / 2, n))  # sparse left
        return np.c_[np.cos(theta), np.sin(theta)] + 0.02 * rng.standard_normal((n, 2))
 
    # three disjoint draws from the SAME distribution:
    #   train       -> fit the subspace / filter (the estimator)
    #   calibration -> set the threshold tau (NOT used to fit the estimator)
    #   test        -> report empirical coverage of genuine on-support points
    Xtr  = sample_circle(800)
    Xcal = sample_circle(400)
    Xte  = sample_circle(400)
 
    est = RudiSupportEstimator(kernel="abel", gamma=2.0, filter="tikhonov",
                               reg=5e-4, n_pivots=200, center=True).fit(Xtr)
 
    # ------------------------------------------------------------------ #
    # All three estimators reuse the SAME fitted pieces (pivots, spectrum,
    # residual Delta).  They differ ONLY in the in-subspace eigenvalue
    # weight applied to a_j(x)^2:
    #
    #   Rudi  : (1 - r_j)^2          squared residual filter  -> distance
    #   GP var: (1 - r_j)^1          linear   residual filter  -> variance
    #   KPCA  : 1[j not in top-m]    hard cut-off             -> reconstruction err
    #
    # In every case the shared out-of-subspace residual Delta(x) is added,
    # so the panels isolate exactly the first-vs-second-power effect.
    # ------------------------------------------------------------------ #
    def score_grid(weight_vec, Xg):
        psi = est._psi(Xg)                                  # (n, r)
        a = (psi - est.mean_) @ est.W_                      # (n, r)
        delta = np.maximum(1.0 - np.einsum("ij,ij->i", psi, psi), 0.0)
        return (a ** 2) @ weight_vec + delta
 
    r_j = est.sigma_ / (est.sigma_ + est.reg)               # Tikhonov r(sigma_j)
    m = int(np.sum(r_j > 0.5))                              # KPCA rank ~ effective
    order = np.argsort(est.sigma_)[::-1]
    hard = np.ones_like(r_j); hard[order[:m]] = 0.0          # (1 - r) for hard cut
 
    panels = [
        ("Rudi  (squared filter, $(1-r)^2$)", (1.0 - r_j) ** 2),
        ("GP posterior variance ($(1-r)^1$)", (1.0 - r_j)),
        (f"KPCA  (hard cut-off, m={m})",       hard),
    ]
 
    # ------------------------------------------------------------------ #
    # SPLIT-CONFORMAL threshold: tau = the k-th smallest CALIBRATION score,
    # k = ceil((n_cal + 1)(1 - alpha)).  This guarantees, marginally, that a
    # fresh on-support point falls inside the estimated support with prob.
    # >= 1 - alpha.  The calibration set is disjoint from the fit data, so
    # the threshold is honest (no train-on-the-boundary optimism).
    # ------------------------------------------------------------------ #
    alpha = 0.0
    def conformal_tau(scores_cal):
        m_cal = len(scores_cal)
        k = int(np.ceil((m_cal + 1) * (1.0 - alpha)))
        k = min(k, m_cal)                                   # clip for small m_cal
        return np.sort(scores_cal)[k - 1]
 
    thresholds = {name: conformal_tau(score_grid(w, Xcal)) for name, w in panels}
 
    # evaluation grid
    g = np.linspace(-1.8, 1.8, 260)
    GX, GY = np.meshgrid(g, g)
    Xg = np.c_[GX.ravel(), GY.ravel()]
 
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, (name, w) in zip(axes, panels):
        Z = score_grid(w, Xg).reshape(GX.shape)
        tau = thresholds[name]
        ax.contourf(GX, GY, Z, levels=30, cmap="viridis")
        ax.contour(GX, GY, Z, levels=[tau], colors="white", linewidths=2.5)
        ax.scatter(Xtr[:, 0], Xtr[:, 1], s=3, c="white", alpha=0.30, linewidths=0)
        ax.scatter(Xcal[:, 0], Xcal[:, 1], s=4, c="orange", alpha=0.35,
                   linewidths=0)
        ax.set_title(name, fontsize=11)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Support calibrated on held-out set (target coverage "
                 f"{1 - alpha:.0%}); white=boundary, orange=calibration pts",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("levelsets.png", dpi=130)
    print("saved levelsets.png\n")
 
    # ------------------------------------------------------------------ #
    # Empirical coverage on the TEST set (should be ~ 1 - alpha = 95%),
    # plus behaviour at the diagnostic points.
    # ------------------------------------------------------------------ #
    pts = {"on dense": [[1., 0.]], "on sparse": [[-1., 0.]],
           "off near": [[1.4, 0.]], "off center": [[0., 0.]]}
    print(f"target coverage = {1 - alpha:.0%}   "
          f"(n_train={len(Xtr)}, n_cal={len(Xcal)}, n_test={len(Xte)})\n")
    for name, w in panels:
        tau = thresholds[name]
        cover = np.mean(score_grid(w, Xte) <= tau)          # on-support coverage
        inside = {k: bool(score_grid(w, np.array(v))[0] <= tau)
                  for k, v in pts.items()}
        print(f"{name:40s} tau={tau:6.4f}  test coverage={cover:5.1%}")
        print(f"{'':40s} probes inside: {inside}")
 
