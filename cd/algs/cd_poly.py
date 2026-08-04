import numpy as np
from scipy.linalg import solve_triangular
from sklearn.kernel_approximation import PolynomialCountSketch

from cd.algs.bases.poly_basis import BasisSpec, ChebyshevBasis, HermiteBasis, MonomialBasis
from cd.utils.plotting import plot_contours


class CDPolynomial:
    def __init__(
        self,
        data: np.ndarray,
        degree: int,
        basis: str = "mon",
        method: str = "qr",
        eps: float = 0.0,
        verbose: bool = False,
        weights: np.ndarray | None = None,
    ):
        data = np.asarray(data)
        self.n_data, self.n_vars = data.shape
        self.deg = degree
        self.verbose = verbose

        assert method in ["chol", "lstsq", "solve", "qr"]
        self.method = method
        self.weights = np.asarray(weights) if weights is not None else None

        bs = BasisSpec(n_vars=self.n_vars, degree=self.deg)
        if basis == "mon":
            self.basis = MonomialBasis(bs)
            self.V = self.basis.transform(data)
        elif basis == "cheb":
            self.basis = ChebyshevBasis(bs)
            self.V = self.basis.transform(data)
        elif basis == "herm":
            self.basis = HermiteBasis(bs)
            self.V = self.basis.transform(data)
        elif basis == "rf":  # more experiments needed
            self.basis = PolynomialCountSketch(
                degree=degree, coef0=1, n_components=100, random_state=0
            )
            self.V = self.basis.fit_transform(data)
        else:
            raise AssertionError("Invalid basis.")

        self.V = np.asarray(self.V)
        self.n_terms = self.V.shape[1]

        if self.weights is not None:
            self.M = (self.V.T @ (self.weights[:, None] * self.V))  + eps * np.eye(self.n_terms)
        else:
            self.M = (self.V.T @ self.V) / self.n_data + eps * np.eye(self.n_terms)

        if self.method == "chol":
            self.L = np.linalg.cholesky(self.M)
        elif self.method == "qr":
            _, self.R = np.linalg.qr(self.V / np.sqrt(self.n_data))

        self.mean = self(data).mean()

        if verbose:
            print(f"Data monomials shape: {self.V.shape}")
            print(f"Moment matrix cond num: {np.linalg.cond(self.M):e}")
            print(f"Empirical mean of CD poly: {self.mean}")

    def __call__(self, z: np.ndarray) -> np.ndarray:
        "Evaluate the CD polynomial at an array of points."
        v = np.asarray(self.basis.transform(z))  # (B, n_terms)

        if self.method == "chol":
            y = solve_triangular(self.L, v.T, lower=True).T  # (B, n_terms)
            return np.einsum("bi,bi->b", y, y)
        elif self.method == "qr":
            y = solve_triangular(self.R.T, v.T, lower=True).T  # (B, n_terms)
            return np.einsum("bi,bi->b", y, y)
        elif self.method == "solve":
            # Gaussian elimination with the moment matrix
            y = np.linalg.solve(self.M, v.T).T  # (B, n_terms)
            return np.einsum("bi,bi->b", v, y)
        elif self.method == "lstsq":
            # Least squares without computing the moment matrix
            y, res, _, _ = np.linalg.lstsq(
                self.V.T / np.sqrt(self.n_data), v.T, rcond=0
            )
            if res.size != 0 and self.verbose:
                print(f"LS had residuals {res}")
            return np.einsum("ib,ib->b", y, y)
        else:
            raise AssertionError("Invalid method.")

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self(z)

    def plot(self, ax, multiplier=1.0, **plot_kwargs):
        """Plot the contours of this CD polynomial."""
        levels = [multiplier * self.mean * 10**i for i in range(10)]
        plot_contours(self, ax, levels=levels, **plot_kwargs)
