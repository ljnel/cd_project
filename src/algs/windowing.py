import logging
import numpy as np

logger = logging.getLogger("cd.algs.windowing")


def estimate_window(x: np.ndarray, period: int = 1, method: str = 'median') -> int:
    """Estimate window length from the dominant FFT frequency.

    Suited for periodic signals where the fundamental frequency
    determines a natural window size.

    x: (n_episodes, seq_len, n_features)
    """
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


def estimate_window_acf(x: np.ndarray, min_window: int = 10) -> int:
    """Estimate window as decorrelation time via ACF zero-crossing.

    Computes the autocorrelation for each episode/feature, averages across
    all of them, and returns the lag of the first zero-crossing.
    Suited for non-periodic signals.

    x: (n_episodes, seq_len, n_features)
    """
    n_episodes, seq_len, n_features = x.shape
    # Mean-subtract per episode/feature
    x_centered = x - x.mean(axis=1, keepdims=True)

    # Compute normalized ACF using FFT for efficiency
    # Pad to avoid circular correlation artifacts
    n_fft = 2 * seq_len
    X_freq = np.fft.rfft(x_centered, n=n_fft, axis=1)
    acf_full = np.fft.irfft(X_freq * np.conj(X_freq), n=n_fft, axis=1)
    # Keep only non-negative lags up to seq_len
    acf_full = acf_full[:, :seq_len, :]
    # Normalize so lag-0 = 1
    acf_full = acf_full / (acf_full[:, 0:1, :] + 1e-12)

    # Average ACF across episodes and features
    mean_acf = acf_full.mean(axis=(0, 2))  # (seq_len,)

    # Find first zero-crossing
    zero_crossings = np.where(mean_acf[1:] <= 0)[0]
    if len(zero_crossings) > 0:
        window = int(zero_crossings[0]) + 1  # +1 because we sliced from index 1
    else:
        # No zero-crossing found: use half the sequence length
        window = seq_len // 2

    window = max(min_window, window)
    logger.debug(f"ACF window estimate: {window}")
    return window
