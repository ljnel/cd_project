from dataclasses import dataclass
from typing import Literal, Optional

import jax.numpy as jnp
import jax.scipy as jsp

from algs.bases import BasisSpec, ChebyshevBasis, MonomialBasis
from utils.plotting import plot_contours, plot_map

Method = Literal["chol", "qr"]
BasisName = Literal["mon", "cheb"]


@dataclass
class CDState:
    """All the arrays needed to evaluate a fitted CD polynomial."""
    # basis info
    basis_name: BasisName
    degree: int
    n_vars: int

    # design / moment info
    n_data: int
    n_terms: int
    eps: float

    # factorization cache (one of these is used depending on `method`)
    method: Method
    L: Optional[jnp.ndarray]  # Cholesky factor of M (lower-triangular)
    R: Optional[jnp.ndarray]  # R from reduced QR of X_bar

    # We re-create the basis from (basis_name, degree, n_vars) when needed.


def _make_basis(basis_name: BasisName, n_vars: int, degree: int):
    """Factory for the basis object (kept small to avoid surprises under JIT)."""
    bs = BasisSpec(n_vars=n_vars, degree=degree)
    if basis_name == "mon":
        return MonomialBasis(bs)
    elif basis_name == "cheb":
        return ChebyshevBasis(bs)
    else:
        raise ValueError(f"Unsupported basis: {basis_name!r}")


def fit_cd(
    data: jnp.ndarray,
    degree: int,
    basis: BasisName = "mon",
    method: Method = "qr",
    eps: float = 0.0,
    verbose: bool = False,
) -> CDState:
    """
    Functional 'fit': returns a CDState that contains everything needed to evaluate P(x).
    Differentiable w.r.t. `data` (provided basis.transform is JAX-only).
    """
    data = jnp.asarray(data)
    n_data, n_vars = data.shape
    basis_impl = _make_basis(basis, n_vars, degree)

    # Feature / design matrix
    X = jnp.asarray(basis_impl.transform(data))  # (n_data, n_terms)
    _, n_terms = X.shape

    # Keep the original contract: we expect n_terms <= n_data.
    # assert (
    #     n_terms <= n_data
    # ), "Require n_terms <= n_data (increase data or reduce degree)."

    # Normalized design and moment matrix
    X_bar = X / jnp.sqrt(n_data)
    M = X_bar.T @ X_bar
    if eps:
        M = M + eps * jnp.eye(n_terms, dtype=M.dtype)

    # Precompute factorization for the chosen method
    if method == "chol":
        L = jnp.linalg.cholesky(M)  # M = L L^T (PD if eps>0)
        R = None
    elif method == "qr":
        # Reduced QR of X_bar: X_bar = Q R, M = R^T R.
        # NOTE: This corresponds to eps == 0 in theory. With eps>0, prefer 'chol'.
        _, R = jnp.linalg.qr(X_bar, mode="reduced")
        L = None
    else:
        raise ValueError(f"Unsupported method: {method!r}")

    if verbose:
        try:
            cond = jnp.linalg.cond(M)
        except Exception:
            cond = jnp.nan
        print(
            f"[fit_cd] X shape={X.shape}, M cond={cond:e}, method={method}, eps={eps}"
        )

    return CDState(
        basis_name=basis,
        degree=degree,
        n_vars=n_vars,
        n_data=n_data,
        n_terms=n_terms,
        eps=eps,
        method=method,
        L=L,
        R=R,
    )


def evaluate_cd(state: CDState, x: jnp.ndarray) -> jnp.ndarray:
    """
    Functional 'evaluate': returns vector P(x) (one scalar per row in x).
    Differentiable w.r.t. x (and w.r.t. state if state came from `fit_cd(data)` inside the trace).
    """
    x = jnp.asarray(x)
    basis_impl = _make_basis(state.basis_name, state.n_vars, state.degree)
    v = jnp.asarray(basis_impl.transform(x))  # (batch, n_terms)

    if state.method == "chol":
        # P(x) = || L^{-T} v ||^2, where M = L L^T
        assert state.L is not None
        y = jsp.linalg.solve_triangular(state.L, v.T, lower=True).T  # (batch, n_terms)
        return jnp.einsum("bi,bi->b", y, y)

    # state.method == "qr"
    # P(x) = || R^{-T} v ||^2 since M = R^T R  (best used with eps=0)
    assert state.R is not None
    y = jsp.linalg.solve_triangular(state.R.T, v.T, lower=True).T  # (batch, n_terms)
    return jnp.einsum("bi,bi->b", y, y)


class CDPolynomial:
    """
    Notes for training:
      • JAX's grad needs a scalar; wrap calls with a reduction at the loss site.
      • For ∂/∂x: grad(lambda xi: evaluate_cd(state, xi[None]).mean())(xi)
      • For ∂/∂data: grad(lambda D: evaluate_cd(fit_cd(D, ...), X_query).mean())(data)
    """

    def __init__(
        self,
        data: jnp.ndarray,
        degree: int,
        basis: BasisName = "mon",
        method: Method = "qr",
        eps: float = 0.0,
        verbose: bool = False,
    ):
        self.deg = degree
        self.verbose = verbose

        assert method in ['chol', 'lstsq', 'solve', 'qr']
        self.method = method

        bs = BasisSpec(n_vars=self.n_vars, degree=self.deg)
        if basis == 'mon':
            self.basis = MonomialBasis(bs)
            self.V = self.basis.transform(data)
        elif basis == 'cheb':
            self.basis = ChebyshevBasis(bs)
            self.V = self.basis.transform(data)
        elif basis == 'rf':  # more experiments needed
            self.basis = PolynomialCountSketch(degree=degree, coef0=1, 
                                               n_components=100, random_state=0)  # ???
            self.V = self.basis.fit_transform(data)
        else:
            raise AssertionError("Invalid basis.")

        self.n_terms = self.V.shape[1]
        #assert self.n_terms <= self.n_data

        self.M = (self.V.T @ self.V) / self.n_data + eps * np.eye(self.n_terms)

        if self.method == 'chol':
            self.L = np.linalg.cholesky(self.M)
        elif self.method == 'qr':
            _, self.R = np.linalg.qr(self.V / np.sqrt(self.n_data))

        self.mean = self(data).mean()
        
        if verbose:
            print(f'Data monomials shape: {self.V.shape}')
            print(f'Moment matrix cond num: {np.linalg.cond(self.M):e}')
            print(f'Empirical mean of CD poly: {self.mean}')

    def __call__(self, z: np.ndarray) -> np.ndarray: 
        "Evaluate the CD polynomial at an array of points."
        v = self.basis.transform(z)  # (B, n_terms)

        if self.method == 'chol':
            y = solve_triangular(self.L, v.T, lower=True).T  # (B, n_terms)
            return np.einsum('bi,bi->b', y, y)
        elif self.method == 'qr':
            y = solve_triangular(self.R.T, v.T, lower=True).T  # (B, n_terms)
            return np.einsum('bi,bi->b', y, y)
        elif self.method == 'solve':
            # Gaussian elimination with the moment matrix
            y = np.linalg.solve(self.M, v.T).T  # (B, n_terms)
            return np.einsum('bi,bi->b', v, y)
        elif self.method == 'lstsq':
            # Least squares without computing the moment matrix
            y, res, _, _ = np.linalg.lstsq(self.V.T / np.sqrt(self.n_data),
                                           v.T, rcond=0)
            if res.size != 0 and self.verbose:
                print(f'LS had residuals {res}')
            return np.einsum('ib,ib->b', y, y)
        
        else:
            raise AssertionError("Invalid method.")

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self(z)

    def plot(self, ax, multiplier=1., **plot_kwargs):
        """
        Plot the contours of this CD polynomial.
        """
        levels = [multiplier * self.mean * 10 ** i for i in range(10)]
        plot_contours(self, ax, levels=levels, **plot_kwargs)
