"""Truncated sequential kernel for time series (Kiraly & Oberhauser, 2019).

Vectorised implementation of the level-L sequential kernel::

    SK_L(x, y) = 1 + Σ_{s,t} K[s,t] · R_L[s,t]

where K[s,t] = k_static(x[s], y[t]) is a user-supplied static kernel
and R_L is built by iterating a reverse-cumulative-sum recurrence L−1 times.

Supports both CPU (numpy) and GPU (torch) backends.  The GPU path avoids
the Python chunking loop by computing the full 4-D static-kernel tensor in
one fused call.

Reference
---------
F. Kiraly, H. Oberhauser. "Kernels for sequentially ordered data."
Journal of Machine Learning Research, 2019.
"""

from collections.abc import Callable

import numpy as np
import torch

from .base import Kernel

# Static kernel: (N, d) × (M, d) → (N, M)
StaticKernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# 4D Gram function: (nx, lx, d) × (ny, ly, d) → (nx, ny, lx, ly)
StaticKernel4DFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── NumPy helpers ───────────────────────────────────────────────────────────

def _cumsum_rev_batch(A: np.ndarray) -> np.ndarray:
    """Reverse cumulative sum along the last two axes."""
    return np.cumsum(
        np.cumsum(A[..., ::-1, ::-1], axis=-2), axis=-1,
    )[..., ::-1, ::-1]


def _seq_kernel_batch_np(
    K_static: np.ndarray,
    level: int,
    normalize: bool,
) -> np.ndarray:
    """Evaluate the sequential kernel on batched static kernel matrices (CPU)."""
    R = np.ones_like(K_static)
    if normalize:
        normfac = K_static.shape[-2] * K_static.shape[-1]
        for _ in range(level - 1):
            R = (1.0 + _cumsum_rev_batch(K_static * R) / normfac) / 2.0
        return (1.0 + (K_static * R).sum(axis=(-2, -1)) / normfac) / 2.0
    else:
        for _ in range(level - 1):
            R = 1.0 + _cumsum_rev_batch(K_static * R)
        return 1.0 + (K_static * R).sum(axis=(-2, -1))


def _static_kernel_4d_generic(
    X: np.ndarray,
    Y: np.ndarray,
    static_kernel: StaticKernelFn,
) -> np.ndarray:
    """Compute K[i, j, s, t] via flattening and a tabular kernel call."""
    nx, lx, d = X.shape
    ny, ly, _ = Y.shape
    X_flat = X.reshape(nx * lx, d)
    Y_flat = Y.reshape(ny * ly, d)
    K_flat = static_kernel(X_flat, Y_flat)
    return K_flat.reshape(nx, lx, ny, ly).transpose(0, 2, 1, 3)


# ── Torch helpers ──────────────────────────────────────────────────────────

def _cumsum_rev_batch_torch(A: torch.Tensor) -> torch.Tensor:
    """Reverse cumulative sum along the last two axes (GPU)."""
    return torch.cumsum(
        torch.cumsum(A.flip(-2, -1), dim=-2), dim=-1,
    ).flip(-2, -1)


def _seq_kernel_batch_torch(
    K_static: torch.Tensor,
    level: int,
    normalize: bool,
) -> torch.Tensor:
    """Evaluate the sequential kernel on batched static kernel matrices (GPU)."""
    R = torch.ones_like(K_static)
    if normalize:
        normfac = K_static.shape[-2] * K_static.shape[-1]
        for _ in range(level - 1):
            R = (1.0 + _cumsum_rev_batch_torch(K_static * R) / normfac) / 2.0
        return (1.0 + (K_static * R).sum(dim=(-2, -1)) / normfac) / 2.0
    else:
        for _ in range(level - 1):
            R = 1.0 + _cumsum_rev_batch_torch(K_static * R)
        return 1.0 + (K_static * R).sum(dim=(-2, -1))


# ── Main class ──────────────────────────────────────────────────────────────

class TruncatedSigKernel(Kernel):
    """Truncated sequential kernel with an arbitrary static kernel.

    Parameters
    ----------
    static_kernel : callable  (N, d) × (M, d) → (N, M)
        A tabular kernel function applied at each timestep (CPU path).
    level : int, default=2
        Truncation level (≥ 1).
    normalize : bool, default=True
        Divide cumulative sums by the number of timestep pairs at each level.
    max_chunk : int, default=30
        Maximum block size for the CPU chunking loop.
    static_kernel_4d : callable, optional
        Direct 4-D computation for the CPU path.
        Signature: (nx, lx, d) × (ny, ly, d) → (nx, ny, lx, ly).
    gpu : bool, default=True
        Use the GPU (torch) backend when CUDA is available.
    gpu_rbf_gamma : float, optional
        If set, enables a fused GPU RBF path that avoids the generic
        static-kernel interface entirely.  Must be set for GPU acceleration.
    gpu_max_chunk : int, default=512
        Maximum number of row paths per GPU chunk (controls VRAM usage).
    """

    def __init__(
        self,
        static_kernel: StaticKernelFn,
        level: int = 2,
        normalize: bool = True,
        max_chunk: int = 30,
        static_kernel_4d: StaticKernel4DFn | None = None,
        gpu: bool = True,
        gpu_rbf_gamma: float | None = None,
        gpu_max_chunk: int = 512,
    ):
        self.static_kernel = static_kernel
        self.level = level
        self.normalize = normalize
        self.max_chunk = max_chunk
        self.gpu_max_chunk = gpu_max_chunk
        self._kernel_4d = (
            static_kernel_4d
            if static_kernel_4d is not None
            else lambda X, Y: _static_kernel_4d_generic(X, Y, static_kernel)
        )

        # GPU setup
        self._device = _get_device()
        self._use_gpu = gpu and self._device.type == "cuda"
        self._gpu_rbf_gamma = gpu_rbf_gamma

        # Cached raw diagonal from the last symmetric __call__ (training)
        self._train_diag_raw: np.ndarray | None = None

    # -- GPU path ------------------------------------------------------------

    def _rbf_4d_torch(
        self, X: torch.Tensor, Y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute K[i,j,s,t] = exp(-γ||X[i,s]-Y[j,t]||²) on GPU."""
        gamma = self._gpu_rbf_gamma
        X_sq = (X * X).sum(dim=-1)                     # (nx, lx)
        Y_sq = (Y * Y).sum(dim=-1)                     # (ny, ly)
        dot = torch.einsum('isd,jtd->ijst', X, Y)      # (nx, ny, lx, ly)
        dist_sq = X_sq[:, None, :, None] + Y_sq[None, :, None, :] - 2.0 * dot
        dist_sq.clamp_(min=0)
        return torch.exp(-gamma * dist_sq)

    def _gram_gpu(
        self,
        x: np.ndarray,
        y: np.ndarray,
        symmetric: bool,
    ) -> np.ndarray:
        """Compute the full raw Gram matrix on GPU, chunking rows."""
        nx, ny = len(x), len(y)
        c = self.gpu_max_chunk

        # Adapt chunk size to available VRAM
        # 4D tensor is (chunk, ny, lx, ly) × 4 bytes; target ~4 GB max
        lx, ly = x.shape[1], y.shape[1]
        bytes_per_row = ny * lx * ly * 4
        max_vram = 4 * 1024**3  # 4 GB
        c = min(c, max(1, int(max_vram / bytes_per_row)))

        # Keep Y on GPU for the duration of the call
        Y_gpu = torch.as_tensor(y, dtype=torch.float32, device=self._device)

        raw = np.empty((nx, ny), dtype=np.float32)
        n_chunks = (nx + c - 1) // c

        for idx, i0 in enumerate(range(0, nx, c)):
            i1 = min(i0 + c, nx)
            X_chunk = torch.as_tensor(
                x[i0:i1], dtype=torch.float32, device=self._device,
            )
            with torch.no_grad():
                K4 = self._rbf_4d_torch(X_chunk, Y_gpu)
                block = _seq_kernel_batch_torch(
                    K4, self.level, self.normalize,
                )
            raw[i0:i1] = block.cpu().numpy()


        if symmetric:
            # Force exact symmetry; floating-point noise can desync upper/lower.
            raw = (raw + raw.T) / 2.0

        return raw.astype(np.float64)

    def _diag_gpu(self, x: np.ndarray) -> np.ndarray:
        """Raw self-kernel diagonal on GPU."""
        n = len(x)
        c = self.gpu_max_chunk
        diag = np.empty(n, dtype=np.float32)
        for i0 in range(0, n, c):
            i1 = min(i0 + c, n)
            X_chunk = torch.as_tensor(
                x[i0:i1], dtype=torch.float32, device=self._device,
            )
            with torch.no_grad():
                # Self-kernel: (chunk, 1, l, l)
                K4 = self._rbf_4d_torch(
                    X_chunk, X_chunk,
                )
                # We only need the diagonal entries K4[i, i, :, :]
                K4_diag = K4[
                    torch.arange(i1 - i0, device=self._device),
                    torch.arange(i1 - i0, device=self._device),
                ]  # (chunk, l, l)
                vals = _seq_kernel_batch_torch(
                    K4_diag, self.level, self.normalize,
                )
            diag[i0:i1] = vals.cpu().numpy()
        return diag.astype(np.float64)

    # -- CPU path (unchanged) ------------------------------------------------

    def _gram_cpu(
        self,
        x: np.ndarray,
        y: np.ndarray,
        symmetric: bool,
    ) -> np.ndarray:
        nx, ny = len(x), len(y)
        c = self.max_chunk
        raw = np.empty((nx, ny))
        for i0 in range(0, nx, c):
            i1 = min(i0 + c, nx)
            j_start = i0 if symmetric else 0
            for j0 in range(j_start, ny, c):
                j1 = min(j0 + c, ny)
                K4 = self._kernel_4d(x[i0:i1], y[j0:j1])
                raw[i0:i1, j0:j1] = _seq_kernel_batch_np(
                    K4, self.level, self.normalize,
                )
                if symmetric and i0 != j0:
                    raw[j0:j1, i0:i1] = raw[i0:i1, j0:j1].T
        return raw

    def _diag_cpu(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        diag = np.empty(n)
        for i in range(n):
            xi = x[i:i + 1]
            K4 = self._kernel_4d(xi, xi)
            diag[i] = _seq_kernel_batch_np(
                K4, self.level, self.normalize,
            ).item()
        return diag

    # -- Kernel interface ----------------------------------------------------

    def __call__(self, x: np.ndarray, y=None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        symmetric = y is None
        y = x if symmetric else np.asarray(y, dtype=np.float64)

        use_gpu = self._use_gpu and self._gpu_rbf_gamma is not None

        raw = (self._gram_gpu(x, y, symmetric) if use_gpu
               else self._gram_cpu(x, y, symmetric))

        # Normalise: K_norm[i,j] = raw[i,j] / sqrt(raw[i,i] · raw[j,j])
        if symmetric:
            raw_diag = np.diag(raw).copy()
            self._train_diag_raw = raw_diag  # cache for cross-kernel calls
            diag = np.sqrt(np.maximum(raw_diag, 1e-12))
            raw /= diag[:, None]
            raw /= diag[None, :]
        else:
            # For cross-kernels, reuse cached training diagonal if available
            if use_gpu:
                diag_x = np.sqrt(np.maximum(self._diag_gpu(x), 1e-12))
            else:
                diag_x = np.sqrt(np.maximum(self._diag_cpu(x), 1e-12))

            if (self._train_diag_raw is not None
                    and len(self._train_diag_raw) == len(y)):
                diag_y = np.sqrt(np.maximum(self._train_diag_raw, 1e-12))
            elif use_gpu:
                diag_y = np.sqrt(np.maximum(self._diag_gpu(y), 1e-12))
            else:
                diag_y = np.sqrt(np.maximum(self._diag_cpu(y), 1e-12))

            raw /= diag_x[:, None]
            raw /= diag_y[None, :]

        np.clip(raw, -1, 1, out=raw)
        return raw

    def diag(self, x: np.ndarray) -> np.ndarray:
        """Normalised diagonal is always 1."""
        return np.ones(len(x))
