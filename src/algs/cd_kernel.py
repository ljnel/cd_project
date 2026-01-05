import numpy as np
from scipy.linalg import solve_triangular

from algs.cd_poly import CDPolynomial


class CDPolynomialKernel(CDPolynomial):
    def __init__(self, data, kernel, method: str = "chol", eps=0.0, verbose=False):
        """
        Initialize the CD polynomial based on data.
        Degree is the degree of the basis (half that of the moment matrix).
        We assume that the data points are represented by their coordinates in
        a basis - e.g. for a function space, the coordinates with respect to
        the Chebyshev basis. Eps is a regularization parameter which ensures
        that the moment matrix is nonsingular.

        Cholesky and Gaussian elimination via the moment matrix are
        implemented for completeness, but aren't numerically accurate.
        Least squares and QR decomposition without the moment matrix are
        accurate, with QR being noticeably faster.

        kernel is a callable from sklearn.metrics.pairwise
        """
        self.data = data
        self.n_data, self.n_vars = data.shape  # number of variables
        self.verbose = verbose

        assert method in ["chol", "solve"]
        self.method = method
        self.kernel = kernel

        self.K = self.kernel(data, data)
        self.M = self.K @ self.K
        self.L = None
        self.mean = self(data).mean()

        if verbose:
            print(f"Kernel matrix cond num: {np.linalg.cond(self.K):e}")
            print(f"Empirical mean of CD poly: {self.mean}")

    def get_features(self, z):
        return self.kernel(self.data, z).T  # N x

    def __call__(self, z):
        k = self.get_features(z).T
        if self.method == "chol":
            # K = LL'
            # w = K^-1k= L^-T(L^-1k)
            L = np.linalg.cholesky(self.K)
            y = solve_triangular(L, k, lower=True)  # (batch, n_monomials)
            w = solve_triangular(L.T, y, lower=False)
            value = np.einsum("ib,ib->b", w, w)

        elif self.method == "solve":
            # Gaussian elimination with the moment matrix
            w = np.linalg.solve(self.K, k)  # (batch, n_monomials)
            value = np.einsum("ib,ib->b", w, w)

        else:
            raise ValueError("Unknown method")

        return self.M.shape[0] * value
