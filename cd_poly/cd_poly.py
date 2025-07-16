from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve_triangular
import numpy as np


class CDPolynomial():

    def __init__(self, data, degree=2, method='qr', eps=0., verbose=False):
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
        """
        self.deg = degree  # highest degree of the monomial basis
        self.n_data, self.n_vars = data.shape  # number of variables
        self.verbose = verbose

        assert method in ['chol', 'lstsq', 'solve', 'qr']
        self.method = method

        self.X = PolynomialFeatures(degree=degree).fit_transform(data)
        _, self.n_monomials = self.X.shape
        assert self.n_monomials <= self.n_data

        self.moments = (self.X.T @ self.X) / self.n_data
        self.moments += eps * np.eye(self.n_monomials, self.n_monomials)

        self.mean = self(data).mean()
        
        if verbose:
            print(f'Data monomials shape: {self.X.shape}')
            print(f'Moment matrix cond num: {np.linalg.cond(self.moments):e}')
            print(f'Empirical mean of CD poly: {self.mean}')

    def __call__(self, z):
        """
        Evaluate the CD polynomial at an array of points
        (also represented by coords).
        """
        poly = PolynomialFeatures(degree=self.deg).fit_transform(z)  # (batch, n_monomials)

        if self.method == 'chol':
            # Cholesky decomposition of the moment matrix
            L = np.linalg.cholesky(self.moments)          
            y = solve_triangular(L, poly.T, lower=True).T  # (batch, n_monomials)
            return np.einsum('bi,bi->b', y, y)

        if self.method == 'solve':
            # Gaussian elimination with the moment matrix
            y = np.linalg.solve(self.moments, poly.T).T  # (batch, n_monomials)
            return np.einsum('bi,bi->b', poly, y)

        if self.method == 'lstsq':
            # Least squares without computing the moment matrix
            y, res, _, _ = np.linalg.lstsq(self.X.T / np.sqrt(self.n_data),
                                           poly.T, rcond=0)
            if res.size != 0 and self.verbose:
                print(f'LS had residuals {res}')
            return np.einsum('ib,ib->b', y, y)

        if self.method == 'qr':
            # QR decomposition without computing the moment matrix
            _, R = np.linalg.qr(self.X / np.sqrt(self.n_data))
            y = solve_triangular(R.T, poly.T, lower=True).T  # (batch, n_monomials)
            return np.einsum('bi,bi->b', y, y)

    def plot(self, ax, multiplier=1., **plot_kwargs):
        """
        Plot the contours of this CD polynomial.
        """
        levels = [multiplier * self.mean * 10 ** i for i in range(10)]
        plot_contours(self, ax, levels=levels, **plot_kwargs)


def plot_contours(f, ax, **plot_kwargs):
    "Make a contour plot of a vectorized function on the given axes."

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = f(points)
    Z = vals.reshape(X.shape)

    contours = ax.contour(X, Y, Z, alpha=0.9, **plot_kwargs)
    ax.clabel(contours)


def plot_level_set(f, alpha, ax):
    "Plot the alpha-level set of a vectorized function on the given axes."

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = f(points)
    Z = vals.reshape(X.shape)

    ax.contour(X, Y, Z, levels=[0., alpha], cmap='Blues', alpha=0.9)


def plot_func(f, ax, **plot_kwargs):
    "Plot a vectorized function on [-1, 1]."
    ts = np.linspace(-1, 1, 100)
    ax.plot(ts, f(ts), **plot_kwargs)


def sample_ball_unif(n_samples, rad, dim=2):
    "Sample uniformly from a Euclidean ball."
    x = np.random.randn(n_samples, dim)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    u = np.random.rand(n_samples, 1)
    r = rad * u ** (1.0 / dim)
    return r * x
