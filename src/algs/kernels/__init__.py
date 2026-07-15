# Kernel implementations for the CD detector.
#
# Base class:
#   Kernel              - Abstract base for all kernels
#
# State / vector kernels (operate on flat state vectors):
#   RBF                 - Radial basis function (Gaussian, squared L2 distance)
#   Laplace             - Laplacian kernel (L1 distance)
#   Abel                - Exponential kernel (L2 distance)
#   Polynomial          - Inhomogeneous polynomial kernel (non-stationary)
#
# Signal kernels (operate on individual time series):
#   PolyFFT             - Polynomial kernel on FFT magnitudes
#   GaussFFT            - Gaussian kernel on FFT magnitudes
#   SigKernel           - Path-signature kernel (PDE-based, via sktime)
#   TruncatedSigKernel  - Truncated path-signature kernel
#   ScatteringKernel    - Wavelet scattering transform + RBF
#   MiniRocketKernel    - RBF on MiniRocket random-conv features
#
# Trajectory-level kernels (operate on collections of trajectories):
#   SpatiotemporalKernel - Composed spatial * temporal kernel
#   SumKernel            - Spatial kernel summed over all timestep pairs
#   RFFMeanKernel        - Mean embedding via Random Fourier Features
#
# `temporal_kernel.py` separately defines `TemporalSumKernel` and
# `ProductKernel` as algebra combinators on TemporalKernel objects
# (used via the `+` and `*` operators); not re-exported here.

from algs.kernels.trajectory_kernels import RFFMeanKernel, SpatiotemporalKernel, SumKernel

from .abel import Abel
from .base import Kernel
from .fft import GaussFFT, PolyFFT
from .laplace import Laplace
from .minirocket import MiniRocketKernel
from .polynomial import Polynomial
from .rbf import RBF
from .scattering import ScatteringKernel
from .sequential import TruncatedSigKernel
from .signature import SigKernel

__all__ = [
    "Kernel",
    "RBF",
    "Laplace",
    "Abel",
    "Polynomial",
    "PolyFFT",
    "GaussFFT",
    "SigKernel",
    "TruncatedSigKernel",
    "ScatteringKernel",
    "MiniRocketKernel",
    "SpatiotemporalKernel",
    "SumKernel",
    "RFFMeanKernel",
]
