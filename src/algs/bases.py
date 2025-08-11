from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import lax

MultiIndex = Tuple[int, ...]


def _validate_shape(X: jnp.ndarray, n_vars: int) -> jnp.ndarray:
    X = jnp.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_vars); got {X.ndim}D.")
    if X.shape[1] != n_vars:
        raise ValueError(f"Expected X.shape[1] == {n_vars}; got {X.shape[1]}.")
    return X


def total_degree_index_set(n_vars: int, degree: int) -> List[MultiIndex]:
    """
    All multi-indices alpha in N^n_vars with |alpha|_1 <= degree, graded-lex order.
    (Python-side list is fine; it’s static for JIT.)
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if n_vars <= 0:
        raise ValueError("n_vars must be >= 1")

    indices: List[MultiIndex] = []

    def rec(prefix: List[int], remaining_vars: int, remaining_deg: int):
        if remaining_vars == 1:
            indices.append(tuple(prefix + [remaining_deg]))
            return
        for k in range(remaining_deg + 1):
            rec(prefix + [k], remaining_vars - 1, remaining_deg - k)

    for t in range(degree + 1):
        rec([], n_vars, t)
    return indices


@dataclass(frozen=True)
class BasisSpec:
    n_vars: int
    degree: int
    # Optional per-variable domain [(a1, b1), ..., (an, bn)].
    domain: Optional[Sequence[Tuple[float, float]]] = None


class Basis(ABC):
    """
    Abstract multivariate polynomial basis up to a given total degree.

    Subclasses define `_eval_1d(k, x, dim)`, evaluating the k-th 1D basis
    on a 1D array `x` for a single variable (dimension `dim`).
    """

    def __init__(self, spec: BasisSpec):
        self.n_vars = spec.n_vars
        self.degree = spec.degree
        self.domain = spec.domain
        self._index_set: List[MultiIndex] = total_degree_index_set(
            self.n_vars, self.degree
        )

    # ---- Hooks for subclasses ----

    @abstractmethod
    def _eval_1d(self, k: int, x: jnp.ndarray, dim: int) -> jnp.ndarray: ...

    def _preprocess_var(self, x: jnp.ndarray, dim: int) -> jnp.ndarray:
        return x

    # ---- Public API ----

    @property
    def index_set(self) -> List[MultiIndex]:
        return self._index_set

    @property
    def n_terms(self) -> int:
        return len(self._index_set)

    def transform(self, X: jnp.ndarray, include_intercept: bool = True) -> jnp.ndarray:
        """
        Return feature matrix Phi: shape (n_samples, n_terms)
        Phi[i, j] = prod_d phi_{alpha_j[d]}( X[i, d] )
        """
        X = _validate_shape(X, self.n_vars)
        n, d = X.shape

        # Per-dimension tables V_d: shape (n, degree+1)
        values: List[jnp.ndarray] = []
        for dim in range(d):
            x = self._preprocess_var(X[:, dim], dim)
            Vd = jnp.stack(
                [self._eval_1d(k, x, dim) for k in range(self.degree + 1)], axis=1
            )
            values.append(Vd)

        # Assemble multivariate products for each multi-index alpha
        cols: List[jnp.ndarray] = []
        for alpha in self._index_set:
            # gather per-dim columns and multiply across dims
            per_dim = [values[dim][:, alpha[dim]] for dim in range(d)]
            v = jnp.prod(jnp.stack(per_dim, axis=1), axis=1)
            cols.append(v)

        Phi = jnp.stack(cols, axis=1)
        if not include_intercept:
            Phi = Phi[:, 1:]  # drop alpha=(0,...,0)
        return Phi

    def basis_names(self, var_names: Optional[Sequence[str]] = None) -> List[str]:
        if var_names is None:
            var_names = [f"x{j}" for j in range(self.n_vars)]
        names = []
        for alpha in self._index_set:
            parts = [f"{var_names[j]}^{a}" for j, a in enumerate(alpha) if a != 0]
            names.append("1" if not parts else " ".join(parts))
        return names

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return (
            f"{cls}(n_vars={self.n_vars}, degree={self.degree}, n_terms={self.n_terms})"
        )


# ------------------------ Concrete bases ------------------------


class MonomialBasis(Basis):
    """Multivariate monomials: x^k per coordinate."""

    def _eval_1d(self, k: int, x: jnp.ndarray, dim: int) -> jnp.ndarray:
        if k == 0:
            return jnp.ones_like(x)
        return jnp.power(x, k)


class MonomialBasisScaled(Basis):
    """Multivariate monomials: x^k per coordinate."""

    def _eval_1d(self, k: int, x: jnp.ndarray, dim: int) -> jnp.ndarray:
        # TODO(FD): use the funtionalities in monomials.py to add the correct scaling factors.
        if k == 0:
            return jnp.ones_like(x)
        return jnp.power(x, k)


class ChebyshevBasis(Basis):
    """
    First-kind Chebyshev polynomials T_k on [-1, 1], with optional affine
    scaling from a per-variable domain (a_j, b_j).
    """

    def _preprocess_var(self, x: jnp.ndarray, dim: int) -> jnp.ndarray:
        if self.domain is None:
            return x
        a, b = self.domain[dim]
        # Map [a, b] -> [-1, 1]
        return 2.0 * (x - a) / (b - a) - 1.0

    def _eval_1d(self, k: int, x: jnp.ndarray, dim: int) -> jnp.ndarray:
        # T_0 = 1, T_1 = x, T_{k+1} = 2x T_k - T_{k-1}
        if k == 0:
            return jnp.ones_like(x)
        if k == 1:
            return x

        T0 = jnp.ones_like(x)
        T1 = x

        def body(i, carry):
            tkm1, tk = carry
            tkp1 = 2.0 * x * tk - tkm1
            return (tk, tkp1)

        # Runs for i = 1, ..., k-1 (inclusive of start, exclusive of stop)
        _, Tk = lax.fori_loop(1, k, body, (T0, T1))
        return Tk
