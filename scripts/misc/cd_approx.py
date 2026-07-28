import matplotlib.pyplot as plt
import numpy as np
from cd.algs.cd_pcs import CDApprox
from scipy.stats import kendalltau, rankdata, spearmanr

from cd.algs.cd_poly import CDPolynomial
from cd.utils.paths import get_output_dir
from cd.utils.plotting import save_plot


def ranks_experiment():
    X_train = np.random.standard_normal((2000, 10))
    X_test  = 2 * np.random.standard_normal((1000,  10)) # larger var to create some outliers

    p_exact  = CDPolynomial(X_train, degree=3)
    p_approx = CDApprox(X_train, degree=3, n_components=100)

    C_exact  = p_exact(X_test)
    C_approx = p_approx(X_test)

    assert C_exact.shape == C_approx.shape, "Mismatched lengths"

    # compare ranks
    r_exact  = rankdata(C_exact,  method="ordinal")
    r_approx = rankdata(C_approx, method="ordinal")
    n = len(r_exact)

    plt.figure(figsize=(6, 6))
    plt.scatter(r_exact, r_approx, s=8, alpha=0.6)
    lim = [1, n]
    plt.plot(lim, lim)                              # identity
    band = 0.05 * n                                 # ±5 % envelope
    plt.plot(lim, [lim[0] + band, lim[1] + band], linestyle="--", color='orange')
    plt.plot(lim, [lim[0] - band, lim[1] - band], linestyle="--", color='orange')

    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("True CD rank")
    plt.ylabel("Approx CD rank")
    plt.title("Rank–Rank comparison")

    rho, _ = spearmanr(C_exact, C_approx)
    tau, _ = kendalltau(C_exact, C_approx)
    plt.text(0.05 * n, 0.9 * n, f"Spearman ρ = {rho:.3f}\nKendall τ = {tau:.3f}")

    plt.tight_layout()
    save_plot(get_output_dir() / "ranks.png")


def contour_experiment():
    # compare contours
    X = np.random.randn(1000, 2)
    p_exact = CDPolynomial(X, degree=5)
    p_approx = CDApprox(X, degree=5, n_components=10)
    plt.figure()
    plt.scatter(*X.T, alpha=.2)
    p_exact.plot(plt.gca(), colors='blue')
    p_approx.plot(plt.gca(), colors='red')
    plt.tight_layout()
    save_plot(get_output_dir() / "contours.png")


def values_experiment():
    # values plot: Binom(10 + 5, 5) = 3003
    dim = 10
    degree = 5
    ks = [100, 400, 1500]
    n_bins = 20

    X_train = np.random.standard_normal((10_000, dim))
    models = {
        k: CDApprox(X_train, n_components=k, degree=degree, random_state=0)
        for k in ks
    }

    X_test = np.random.standard_normal((10_000, dim))
    radii = np.linalg.norm(X_test, axis=1)

    bin_edges = np.linspace(radii.min(), radii.max() / 2, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # has length n_bins

    mean_scores = {k: np.full(n_bins, np.nan) for k in ks}

    for k, model in models.items():
        vals = model(X_test)
        for i in range(n_bins):
            mask = (radii >= bin_edges[i]) & (radii <= bin_edges[i + 1])
            if mask.any():
                mean_scores[k][i] = vals[mask].mean()

    mean_scores_true = []
    p = CDPolynomial(X_train, degree=degree)
    vals = p(X_test)
    for i in range(n_bins):
        mask = (radii >= bin_edges[i]) & (radii <= bin_edges[i + 1])
        if mask.any():
            score = vals[mask].mean()
            mean_scores_true.append(score)

    plt.figure(figsize=(7, 5))
    for k in ks:
        plt.plot(bin_centers, mean_scores[k], label=f"n_components={k}")
    plt.plot(bin_centers, mean_scores_true, label='True')

    plt.xlabel(r"Distance from origin  $\|x\|_2$")
    plt.ylabel("Average approximate CD value")
    plt.title(f"Approximations to deg {degree} CD poly on " + r"$\mathcal{N}(0, I_{10})$)")
    plt.legend(title="n_components", frameon=False)
    plt.tight_layout()
    save_plot(get_output_dir() / "radius_val.png")


if __name__ == "__main__":
    #ranks_experiment()
    #contour_experiment()
    values_experiment()