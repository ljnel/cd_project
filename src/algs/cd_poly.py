from algs.bases import *
import jax
import jax.numpy as jnp
from utils.plotting import plot_contours
from sklearn.kernel_approximation import PolynomialCountSketch as PCS


class CDPolynomial():

    def __init__(self, data, degree: int, basis='mon',
                 method='qr', eps=0., verbose=False, n_components=100):
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

        assert method in ['chol', 'lstsq', 'solve', 'qr']
        self.method = method

        assert basis in ['mon', 'cheb', 'rf']
        bs = BasisSpec(n_vars=self.n_vars, degree=degree)
        if basis == 'mon':
            self.basis = MonomialBasis(bs)
            self.X = self.basis.transform(data)
        elif basis == 'cheb':
            self.basis = ChebyshevBasis(bs)
            self.X = self.basis.transform(data)
        elif basis == 'rf':
            assert n_components is not None
            # NB: random state should probably be dealt with somehow
            self.basis = PCS(degree=degree, n_components=n_components, coef0=1., random_state=0)
            self.X = self.basis.fit_transform(data)

        _, self.n_terms = self.X.shape
        assert self.n_terms <= self.n_data

        self.M = (self.X.T @ self.X) / self.n_data
        self.M += eps * jnp.eye(self.n_terms)

        self.mean = self(data).mean()
        
        if verbose:
            print(f'Feature matrix shape: {self.X.shape}')
            print(f'Moment matrix cond num: {jnp.linalg.cond(self.M):e}')
            print(f'Empirical mean of CD poly: {self.mean}')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the CD polynomial at an array of points
        (also represented by coords).
        """
        v = self.basis.transform(x)  # (batch, n_terms)

        if self.method == 'chol':
            # Cholesky decomposition of the moment matrix
            L = jnp.linalg.cholesky(self.M)          
            y = jax.scipy.linalg.solve_triangular(L, v.T, lower=True).T  # (batch, n_terms)
            return jnp.einsum('bi,bi->b', y, y)

        if self.method == 'solve':
            # Gaussian elimination with the moment matrix
            y = jnp.linalg.solve(self.M, v.T).T  # (batch, n_terms)
            return jnp.einsum('bi,bi->b', v, y)

        if self.method == 'lstsq':
            # Least squares without computing the moment matrix
            y, res, _, _ = jnp.linalg.lstsq(self.X.T / jnp.sqrt(self.n_data),
                                           v.T, rcond=0)
            if res.size != 0 and self.verbose:
                print(f'LS had residuals {res}')
            return jnp.einsum('ib,ib->b', y, y)

        if self.method == 'qr':
            # QR decomposition without computing the moment matrix
            _, R = jnp.linalg.qr(self.X / jnp.sqrt(self.n_data))
            y = jax.scipy.linalg.solve_triangular(R.T, v.T, lower=True).T  # (batch, n_terms)
            return jnp.einsum('bi,bi->b', y, y)

    def plot(self, ax, multiplier=1., **plot_kwargs):
        """
        Plot the contours of this CD polynomial.
        """
        levels = [multiplier * self.mean * 10 ** i for i in range(10)]
        return plot_contours(self, ax, levels=levels, **plot_kwargs)
