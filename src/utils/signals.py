import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

logger = logging.getLogger("cd.utils.signals")


def estimate_freq(x: np.ndarray, fs: float) -> float:
    """
    Estimate the fundamental frequency of a (real) periodic signal using the FFT.

    x : signal samples (1D array)
    fs: sampling frequency (Hz)
    """
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    magnitudes = np.abs(X)

    assert x.ndim == 1

    if len(magnitudes) > 1:  # ignore 0 Hz
        magnitudes[0] = 0
    return freqs[np.argmax(magnitudes)]


def analyze(x: np.ndarray, max_f=None, fs: float = 1):
    "Analyze a scalar signal."
    dt = 1.0 / fs
    t = np.arange(0, len(x) * dt, step=dt)
    est_freq = estimate_freq(x, fs)
    logger.debug(f'est. freq.: {est_freq}')

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].plot(t, x)
    ax[0].set_xlabel('s')
    ax[0].set_title('x')

    ax[1].plot(t, recon_signal(x, n_harmonics=20))
    ax[1].set_xlabel('s')
    ax[1].set_title('reconstruction')

    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x), dt)
    mag = np.abs(X)
    mag[0] = 0  # ignore
    ax[2].stem(freq, mag)
    if max_f is not None:
        ax[2].set_xlim(0, freq[freq <= max_f].max())
    ax[2].set_xlabel('freq (Hz)')
    ax[2].set_title('|X|')


def recon_signal(x: np.ndarray, n_harmonics: int) -> np.ndarray:
    """
    Reconstruct a real-valued signal from its rFFT.
    """
    X = np.fft.rfft(x)

    X_trunc = np.zeros_like(X)
    X_trunc[:n_harmonics] = X[:n_harmonics]

    x_recon = np.fft.irfft(X_trunc, n=len(x))
    return x_recon


def spectral_entropy(x, axis=1, trunc=None, eps=1e-12):
    X = np.fft.rfft(x, axis=axis)
    if trunc is not None:
        X = np.take(X, indices=range(trunc), axis=axis)
    P = np.abs(X)**2
    P_sum = np.sum(P, axis=axis, keepdims=True)
    P_norm = P / (P_sum + eps)
    H = -np.sum(P_norm * np.log(P_norm + eps), axis=axis)
    H /= np.log(P.shape[axis])
    return H


def estimate_period_acf(x: np.ndarray, min_lag: int = 2) -> int:
    """Estimate dominant period from the first peak of the autocorrelation.

    Computes the ACF for each episode/feature via FFT, averages across
    all of them, and returns the lag of the first local maximum.

    Parameters
    ----------
    x : ndarray of shape (N, T, D)
        Trajectory data.
    min_lag : int
        Minimum lag to consider (avoids the trivial peak at lag 0).

    Returns
    -------
    period : int
        Estimated dominant period in time steps.
    """
    N, T, D = x.shape
    x_centered = x - x.mean(axis=1, keepdims=True)

    n_fft = 2 * T
    X_freq = np.fft.rfft(x_centered, n=n_fft, axis=1)
    acf_full = np.fft.irfft(X_freq * np.conj(X_freq), n=n_fft, axis=1)
    acf_full = acf_full[:, :T, :]
    acf_full = acf_full / (acf_full[:, 0:1, :] + 1e-12)

    mean_acf = acf_full.mean(axis=(0, 2))

    # Find first local maximum after min_lag
    for i in range(min_lag, T - 1):
        if mean_acf[i] >= mean_acf[i - 1] and mean_acf[i] >= mean_acf[i + 1]:
            return i

    # No peak found — signal is not periodic
    return T


def low_pass(x: np.ndarray, alpha: float) -> np.ndarray:
    # x: (time, channel)
    # y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    b = [alpha]
    a = [1, -(1 - alpha)]
    if x.ndim > 1:
        y = lfilter(b, a, x, axis=-2)  # channel last
    else:
        y = lfilter(b, a, x)
    return y
