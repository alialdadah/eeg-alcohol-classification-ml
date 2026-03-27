"""
preprocessing.py
----------------
Signal processing utilities for EEG data.

Design decisions
~~~~~~~~~~~~~~~~
* Welch's method is used for power spectral density (PSD) estimation instead of
  a raw FFT magnitude.  Welch averages overlapping periodograms, which reduces
  variance and gives a more stable band-power estimate — especially important for
  the short (1 s) epochs in this dataset.

* Four canonical EEG frequency bands are extracted.  Alpha (8–13 Hz) is the
  primary band of interest given the literature on alpha-band suppression in
  chronic alcohol use disorders, but theta and beta are included to give the
  classifier more discriminative signal without hand-picking one band.

* No spatial filtering (e.g., ICA, CAR) is applied here because the formatted
  CSV is already re-referenced to average reference in the preprocessing notebook.
  If you start from raw files, add a bandpass filter (0.5–32 Hz) and a 50 Hz
  notch filter before calling these functions.
"""

import numpy as np
from scipy.signal import welch


# ── Frequency band definitions ─────────────────────────────────────────────
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}


def compute_band_power(
    signal: np.ndarray,
    fs: int = 256,
    fmin: float = 8.0,
    fmax: float = 13.0,
    nperseg: int | None = None,
) -> float:
    """
    Estimate mean power in a frequency band using Welch's method.

    Parameters
    ----------
    signal : 1-D array of float
        Time-domain EEG signal (one electrode, one trial).
    fs : int
        Sampling frequency in Hz.
    fmin, fmax : float
        Lower and upper band edges in Hz (inclusive).
    nperseg : int or None
        Length of each Welch segment.  Defaults to min(fs, len(signal)) so that
        frequency resolution is at least 1 Hz.

    Returns
    -------
    float
        Mean power spectral density (μV²/Hz) within [fmin, fmax].
    """
    signal = np.asarray(signal, dtype=float)
    if nperseg is None:
        nperseg = min(fs, len(signal))

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not band_mask.any():
        return 0.0
    return float(np.mean(psd[band_mask]))


def compute_all_band_powers(
    signal: np.ndarray,
    fs: int = 256,
) -> dict[str, float]:
    """
    Return a dict of band-name → mean PSD for every band in FREQ_BANDS.

    Parameters
    ----------
    signal : 1-D array of float
    fs : int

    Returns
    -------
    dict[str, float]
    """
    return {
        band: compute_band_power(signal, fs=fs, fmin=lo, fmax=hi)
        for band, (lo, hi) in FREQ_BANDS.items()
    }
