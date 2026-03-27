"""
preprocessing.py
----------------
Signal processing utilities for EEG data.

Two feature families are supported:

  1. Band-power features (frequency domain)
     Welch PSD mean in delta / theta / alpha / beta bands.
     Used for any condition (typically S1 obj for a clean single-stimulus design).

  2. P300 ERP features (time domain)
     Peak amplitude, mean amplitude, peak latency, and area under the curve
     in the 250–500 ms post-stimulus window.
     Requires the S2 nomatch (oddball) condition, which reliably elicits P300.

Design notes
~~~~~~~~~~~~
* Welch's method is used instead of raw FFT magnitude because it averages
  overlapping periodograms, reducing spectral variance on short (1 s) epochs.

* P300 features are extracted without baseline correction because the formatted
  CSV epochs start at stimulus onset (t = 0) with no pre-stimulus samples.
  The 250–500 ms window is used directly.  This is standard practice when a
  pre-stimulus baseline is unavailable.

* No spatial filtering (ICA, CAR) is applied here.  The formatted CSV is already
  average-referenced in the preprocessing notebook.
"""

import numpy as np
from scipy.signal import welch


# ── Band-power definitions ─────────────────────────────────────────────────

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
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
        Welch segment length.  Defaults to min(fs, len(signal)).

    Returns
    -------
    float
        Mean PSD (μV²/Hz) within [fmin, fmax].
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
    Return band-name → mean PSD for every band in FREQ_BANDS.
    """
    return {
        band: compute_band_power(signal, fs=fs, fmin=lo, fmax=hi)
        for band, (lo, hi) in FREQ_BANDS.items()
    }


# ── P300 ERP definitions ───────────────────────────────────────────────────

# P300 window: 250–500 ms post-stimulus at 256 Hz
P300_WINDOW_MS: tuple[int, int] = (250, 500)

# Core parietal / central-midline electrodes where P300 is maximal.
# These are the electrodes the original project incorrectly removed.
P300_ELECTRODES: list[str] = ["CZ", "PZ", "P3", "P4", "POZ"]

# Ordered feature names produced by extract_p300_features()
P300_FEATURE_NAMES: list[str] = ["mean_amp", "peak_amp", "peak_lat_ms", "auc"]


def extract_p300_features(
    signal: np.ndarray,
    fs: int = 256,
    window_ms: tuple[int, int] = P300_WINDOW_MS,
) -> dict[str, float]:
    """
    Extract four P300 ERP features from a single electrode's time series.

    The P300 is a positive voltage deflection peaking ~300 ms after an
    unexpected (oddball) stimulus.  It is most prominent at parietal and
    central-midline scalp sites and is reliably reduced in individuals with
    alcohol use disorder (Begleiter & Porjesz, 1999).

    Parameters
    ----------
    signal : 1-D array of float
        Full 256-sample EEG epoch starting at stimulus onset (t = 0).
    fs : int
        Sampling frequency in Hz.
    window_ms : (int, int)
        Start and end of the P300 analysis window in milliseconds.
        Default is (250, 500) ms, which brackets the expected P300 peak.

    Returns
    -------
    dict with keys:
        mean_amp     : mean amplitude in the window (μV)
        peak_amp     : maximum amplitude in the window (μV)
        peak_lat_ms  : latency of the peak in milliseconds
        auc          : area under the waveform (trapezoidal rule)
    """
    signal = np.asarray(signal, dtype=float)

    start_sample = int(window_ms[0] * fs / 1000)
    end_sample   = int(window_ms[1] * fs / 1000)
    window_sig   = signal[start_sample:end_sample]

    if len(window_sig) == 0:
        return {k: 0.0 for k in P300_FEATURE_NAMES}

    peak_idx = int(np.argmax(window_sig))
    peak_lat_ms = (start_sample + peak_idx) / fs * 1000

    return {
        "mean_amp":    float(np.mean(window_sig)),
        "peak_amp":    float(window_sig[peak_idx]),
        "peak_lat_ms": float(peak_lat_ms),
        "auc":         float(np.trapezoid(window_sig)),
    }
