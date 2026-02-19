"""
Code for computing with arbitrary polynomial bases that can be used with numpy or pytorch.
Right now, only monomial basis and chebyshev basis are implemented.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

Tensor = torch.Tensor
MultiIndex = Tuple[int, ...]

def _validate_shape(X: Tensor, n_vars: int) -> Tensor:
    X = torch.as_tensor(X, dtype=torch.get_default_dtype())
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_vars); got {X.ndim}D.")
    if X.shape[1] != n_vars:
        raise ValueError(f"Expected X.shape[1] == {n_vars}; got {X.shape[1]}.")
    return X


def total_degree_index_set(n_vars: int, degree: int) -> List[MultiIndex]:
    """
    All multi-indices in n_vars variables with sum <= degree, in graded-lex order.
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
    on a 1D tensor `x` for a single variable (dimension `dim`).
    """

    def __init__(self, spec: BasisSpec):
        self.n_vars = spec.n_vars
        self.degree = spec.degree
        self.domain = spec.domain
        self._index_set: List[MultiIndex] = total_degree_index_set(self.n_vars, self.degree)

    # ---- Hooks for subclasses ----

    @abstractmethod
    def _eval_1d(self, k: int, x: Tensor, dim: int) -> Tensor:
        ...

    def _preprocess_var(self, x: Tensor, dim: int) -> Tensor:
        return x

    # ---- Public API ----

    @property
    def index_set(self) -> List[MultiIndex]:
        return self._index_set

    @property
    def n_terms(self) -> int:
        return len(self._index_set)

    def transform(self, X: Tensor, include_intercept: bool = True) -> Tensor:
        """
        Return feature matrix Phi: shape (n_samples, n_terms)
        Phi[i, j] = prod_d phi_{alpha_j[d]}( X[i, d] )
        """
        X = _validate_shape(X, self.n_vars)
        n, d = X.shape

        # Per-dimension tables V_d: shape (n, degree+1)
        values: List[Tensor] = []
        for dim in range(d):
            x = self._preprocess_var(X[:, dim], dim)
            Vd = torch.stack(
                [self._eval_1d(k, x, dim) for k in range(self.degree + 1)],
                dim=1,
            )  # (n, degree+1)
            values.append(Vd)

        # Assemble multivariate products for each multi-index alpha
        cols: List[Tensor] = []
        for alpha in self._index_set:
            per_dim = [values[dim][:, alpha[dim]] for dim in range(d)]  # each (n,)
            v = torch.prod(torch.stack(per_dim, dim=1), dim=1)  # (n,)
            cols.append(v)

        Phi = torch.stack(cols, dim=1)  # (n, n_terms)
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
        return f"{cls}(n_vars={self.n_vars}, degree={self.degree}, n_terms={self.n_terms})"


# ------------------------ Concrete bases ------------------------

class MonomialBasis(Basis):
    """Multivariate monomials: x^k per coordinate."""
    def _eval_1d(self, k: int, x: Tensor, dim: int) -> Tensor:
        if k == 0:
            return torch.ones_like(x)
        return x.pow(k)


class ChebyshevBasis(Basis):
    """
    First-kind Chebyshev polynomials T_k on [-1, 1], with optional affine
    scaling from a per-variable domain (a_j, b_j).
    """
    def _preprocess_var(self, x: Tensor, dim: int) -> Tensor:
        if self.domain is None:
            return x
        a, b = self.domain[dim]
        a_t = torch.as_tensor(a, dtype=x.dtype, device=x.device)
        b_t = torch.as_tensor(b, dtype=x.dtype, device=x.device)
        return 2.0 * (x - a_t) / (b_t - a_t) - 1.0  # map [a, b] -> [-1, 1]

    def _eval_1d(self, k: int, x: Tensor, dim: int) -> Tensor:
        # T_0 = 1, T_1 = x, T_{k+1} = 2x T_k - T_{k-1}
        if k == 0:
            return torch.ones_like(x)
        if k == 1:
            return x

        Tkm1 = torch.ones_like(x)
        Tk = x
        for _ in range(1, k):  # runs for i = 1, ..., k-1
            Tkp1 = 2.0 * x * Tk - Tkm1
            Tkm1, Tk = Tk, Tkp1
        return Tk