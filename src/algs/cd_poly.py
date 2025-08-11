import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.kernel_approximation import PolynomialCountSketch as PCS

from algs.bases import *
from algs.monomials import get_monomial
from utils.plotting import plot_contours, plot_map


class CDPolynomial:

    def __init__(
        self,
        data,
        degree: int,
        basis="mon",
        method="qr",
        eps=0.0,
        verbose=False,
        n_components=100,
    ):
        """
        Initialize the CD polynomial based on data.
        Degree is the degree of the basis (half that of the moment matrix).
        Basis can be 'mon' (standard monomial basis), 'cheb' (Chebyshev), or 'rf' (random features).
        In the case of 'rf', n_components specifies how many random features to use.
        Eps is a regularization parameter which ensures that the moment matrix is nonsingular.

        Cholesky and Gaussian elimination on the moment matrix are
        implemented for completeness, but Least Squares or QR on the design
        matrix seems to be better.
        """
        self.deg = degree  # highest degree of the monomial basis
        self.n_data, self.n_vars = data.shape  # number of variables
        self.verbose = verbose

        assert method in ["chol", "lstsq", "solve", "qr"]
        self.method = method

        assert basis in ["mon", "mons", "cheb", "rf"]
        bs = BasisSpec(n_vars=self.n_vars, degree=degree)
        if basis == "mon":
            self.basis = MonomialBasis(bs)
            self.X = self.basis.transform(data)
        elif basis == "cheb":
            self.basis = ChebyshevBasis(bs)
            self.X = self.basis.transform(data)
        elif basis == "rf":
            assert n_components is not None
            # NB: random state should probably be dealt with somehow
            self.basis = PCS(
                degree=degree, n_components=n_components, coef0=1, random_state=0
            )
            self.X = self.basis.fit_transform(data)
        elif basis == "mons":
            # TODO(FD) will be cleaner to move to this but needs to be implemented.
            # self.basis = MonomialBasisScaled(bs)
            # self.X = self.basis.transform(data)
            self.X = self.get_features(data)  # K x n_data
            self.basis = None

        _, self.n_terms = self.X.shape
        # assert self.n_terms <= self.n_data

        self.M = (self.X.T @ self.X) / self.n_data
        self.M += eps * jnp.eye(self.n_terms)

        self.mean = self(data).mean()

        if verbose:
            print(f"Feature matrix shape: {self.X.shape}")
            print(f"Moment matrix cond num: {jnp.linalg.cond(self.M):e}")
            print(f"Empirical mean of CD poly: {self.mean}")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the CD polynomial at an array of points
        (also represented by coords).
        """
        if self.basis is None:
            v = self.get_features(x)  # D y N
        else:
            v = self.basis.transform(x)  # type: ignore # (batch, n_terms)

        if self.method == "chol":
            # Cholesky decomposition of the moment matrix
            L = jnp.linalg.cholesky(self.M)
            y = jax.scipy.linalg.solve_triangular(
                L, v.T, lower=True
            ).T  # (batch, n_terms)
            return jnp.einsum("bi,bi->b", y, y)

        elif self.method == "solve":
            # Gaussian elimination with the moment matrix
            y = jnp.linalg.solve(self.M, v.T).T  # (batch, n_terms)
            return jnp.einsum("bi,bi->b", v, y)

        elif self.method == "lstsq":
            # Least squares without computing the moment matrix
            y, res, _, _ = jnp.linalg.lstsq(
                self.X.T / jnp.sqrt(self.n_data), v.T, rcond=0
            )
            if res.size != 0 and self.verbose:
                print(f"LS had residuals {res}")
            return jnp.einsum("ib,ib->b", y, y)

        elif self.method == "qr":
            # QR decomposition without computing the moment matrix
            _, R = jnp.linalg.qr(self.X / jnp.sqrt(self.n_data))
            y = jax.scipy.linalg.solve_triangular(
                R.T, v.T, lower=True
            ).T  # (batch, n_terms)
            return jnp.einsum("bi,bi->b", y, y)
        else:
            raise ValueError("Unknown method")

    def get_features(self, z):
        return np.vstack([get_monomial(xi, d=self.deg) for xi in z])

    def plot(self, ax, multiplier=1.0, **plot_kwargs):
        """
        Plot the contours of this CD polynomial.
        """
        levels = [multiplier * self.mean * 10**i for i in range(10)]
        return plot_contours(self, ax, levels=levels, **plot_kwargs)

    def plot_map(self, ax, **plot_kwargs):
        """
        Plot the contours of this CD polynomial.
        """
        return plot_map(self, ax, **plot_kwargs)


class CDPolynomialKernel(CDPolynomial):
    def __init__(self, data, kernel, method="qr", eps=0.0, verbose=False):
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

        k = self.get_features(z).T  # D y N
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

    def plot(self, ax, multiplier=1.0, **plot_kwargs):
        """
        Plot the contours of this CD polynomial.
        """
        # levels must be increasing
        levels = multiplier * self.mean * np.logspace(-3, 2, 10)
        return plot_contours(self, ax, levels=levels, **plot_kwargs)
