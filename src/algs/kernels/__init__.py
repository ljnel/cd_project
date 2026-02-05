# Kernel implementations for the CD detector
#
# Base class:
#   Kernel - Abstract base class for all kernels
#
# Vector kernels:
#   RBF - Radial basis function (Gaussian) kernel
#
# Path kernels:
#   PolyFFT - Polynomial kernel on FFT magnitudes
#   GaussFFT - Gaussian kernel on FFT magnitudes
#   SigKernel - Signature kernel (PDE-based, via sktime)
#   ScatteringKernel - Wavelet scattering transform + RBF

from .base import Kernel
from .rbf import RBF
from .fft import PolyFFT, GaussFFT
from .signature import SigKernel
from .scattering import ScatteringKernel

__all__ = [
    "Kernel",
    "RBF",
    "PolyFFT",
    "GaussFFT",
    "SigKernel",
    "ScatteringKernel",
]
