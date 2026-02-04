import logging
import numpy as np
import matplotlib.pyplot as plt
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


def estimate_window(x: np.ndarray, period: int = 1, method: str = 'median') -> int:
    steps = x.shape[1]
    fft_magnitudes = np.abs(np.fft.rfft(x, axis=1))
    fft_magnitudes[:, 0, :] = 0
    
    peak_indices = np.argmax(fft_magnitudes, axis=1)
    peak_indices[peak_indices == 0] = 1
    
    frequencies = peak_indices / steps
    
    if method == 'mean':
        avg_frequency = np.mean(frequencies)
    else:
        avg_frequency = np.median(frequencies)
        
    # Convert average frequency back to total window length
    # Period = 1 / Frequency
    estimated_window = period / avg_frequency
    window = int(round(estimated_window))
    logger.debug(f"periods={period}, freq={avg_frequency:.4f} ({method}) → window={window}")

    return window

def analyze(x: np.ndarray, max_f=None, fs: float = 1):
    "Analyze a scalar signal."
    dt = 1.0 / fs
    t = np.arange(0, len(x) * dt, step=dt)
    est_freq = estimate_freq(x, fs)
    print(f'est. freq.: {est_freq}')

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
