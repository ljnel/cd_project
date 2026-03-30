from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from sigkerax.solver import FiniteDifferenceSolver

# Enable float64 in JAX for numerical stability
jax.config.update("jax_enable_x64", True)

from utils.misc import median_heuristic

from .base import Kernel


def _patch_sigkerax_solver():
    """Patch sigkerax's FiniteDifferenceSolver for JAX >=0.4 compatibility.

    sigkerax 0.2.1 marks `p` as a static arg in _solution_diag_update, but
    `p` is a loop variable (tracer) inside fori_loop.  Replacing the decorator
    with static_argnums=(0,) only fixes this — `p` is only used in jnp.where
    which handles tracers fine.
    """
    # Only patch once
    if getattr(FiniteDifferenceSolver, '_patched', False):
        return

    # Unwrap the original function from jit
    orig = FiniteDifferenceSolver._solution_diag_update
    while hasattr(orig, '__wrapped__'):
        orig = orig.__wrapped__

    # Re-wrap with corrected static_argnums (self only)
    FiniteDifferenceSolver._solution_diag_update = partial(
        jax.jit, static_argnums=(0,)
    )(orig)
    FiniteDifferenceSolver._patched = True


_patch_sigkerax_solver()


class SigKernel(Kernel):
    """
    Signature kernel with RBF static kernel (sigkerax / JAX backend).

    Computes the signature kernel via PDE-based method on GPU.
    The signature kernel compares paths via their path signatures —
    the canonical feature map from rough path theory.

    Parameters
    ----------
    gamma : float or "median", default="median"
        Bandwidth for the static RBF kernel applied at each timestep.
        - If "median" (default), computed from training data using the median
          heuristic on flattened signal representations.
        - If a float, used directly.
    """

    def __init__(self, gamma: float | Literal["median"] = "median"):
        self._gamma_param = gamma
        self._gamma: float | None = gamma if isinstance(gamma, (int, float)) else None
        self._solver = None

        if self._gamma is not None:
            self._init_kernel()

    def _init_kernel(self):
        """Initialize the sigkerax PDE solver with the current gamma."""
        # sigkerax RBF uses scale = 1 / sqrt(2 * gamma)
        scale = float(1.0 / np.sqrt(2.0 * self._gamma))
        self._solver = FiniteDifferenceSolver(
            static_kernel_kind="rbf",
            scale=scale,
        )

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            raise ValueError(
                f"Kernel not fitted. Call fit(X) first when using gamma='{self._gamma_param}'."
            )
        return self._gamma

    def fit(self, X: np.ndarray) -> "SigKernel":
        """
        Learn gamma from training data if using a heuristic.

        Parameters
        ----------
        X : np.ndarray of shape (m, n, d)
            Training signals: m signals of length n with d channels.

        Returns
        -------
        self : SigKernel
        """
        if isinstance(self._gamma_param, (int, float)):
            if self._solver is None:
                self._init_kernel()
        elif self._gamma_param == "median":
            from sklearn.metrics.pairwise import euclidean_distances
            X = np.asarray(X)
            m, n, d = X.shape
            all_dists = []
            for t in range(n):
                dists = euclidean_distances(X[:, t, :])
                all_dists.append(dists[np.triu_indices(m, k=1)])
            self._gamma = median_heuristic(np.concatenate(all_dists))
            self._init_kernel()
        else:
            raise ValueError(f"Unknown gamma heuristic: '{self._gamma_param}'")

        # Cache self-kernel diagonal for normalization during predict
        self._train_diag_sqrt = np.sqrt(np.maximum(
            self._raw_diag(jnp.asarray(X, dtype=jnp.float64)), 1e-12,
        ))

        return self

    def _raw(self, x: jnp.ndarray, y: jnp.ndarray) -> np.ndarray:
        """Compute raw (unnormalized) kernel matrix."""
        K = self._solver.solve(x, y)
        return np.array(K[..., 0])  # drop state_space_dim axis

    def _raw_diag(self, x: jnp.ndarray, batch_size: int = 100) -> np.ndarray:
        """Compute diagonal of raw kernel matrix in batches."""
        n = x.shape[0]
        diag = np.empty(n)
        for start in range(0, n, batch_size):
            chunk = x[start:start + batch_size]
            K_chunk = self._solver.solve(chunk, chunk)
            diag[start:start + batch_size] = np.diag(np.array(K_chunk[..., 0]))
        return diag

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        if self._solver is None:
            raise ValueError("Kernel not initialized. Call fit(X) first.")
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        if y is None:
            K = self._raw(x_jax, x_jax)
            diag = np.sqrt(np.maximum(np.diag(K), 1e-12))
            K /= diag[:, None]
            K /= diag[None, :]
            np.clip(K, -1, 1, out=K)
            return K
        y_jax = jnp.asarray(y, dtype=jnp.float64)
        K = self._raw(x_jax, y_jax)
        dx = np.sqrt(np.maximum(self._raw_diag(x_jax), 1e-12))
        # Use cached training diagonal if available and sizes match
        if hasattr(self, '_train_diag_sqrt') and len(self._train_diag_sqrt) == y.shape[0]:
            dy = self._train_diag_sqrt
        else:
            dy = np.sqrt(np.maximum(self._raw_diag(y_jax), 1e-12))
        K /= dx[:, None]
        K /= dy[None, :]
        np.clip(K, -1, 1, out=K)
        return K

    def diag(self, x: np.ndarray) -> np.ndarray:
        """Normalized diagonal is always 1."""
        if self._solver is None:
            raise ValueError("Kernel not initialized. Call fit(X) first.")
        return np.ones(x.shape[0])
