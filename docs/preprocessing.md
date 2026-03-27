# `src/preprocessing.py`

## Purpose

Computes frequency-band power from raw EEG time-series signals. This is the core signal processing layer — it converts a 256-sample voltage waveform into a set of scalar power values, one per frequency band.

---

## Why Welch's method instead of raw FFT

The original project code used raw FFT magnitude on 1-second epochs. The problem: a single raw FFT of a short epoch has high spectral variance — the estimate is noisy and unstable.

**Welch's method** splits the signal into overlapping segments, computes the periodogram of each, and averages them. This reduces variance at the cost of some frequency resolution — a worthwhile trade for 1-second EEG epochs where stability matters more than resolution.

At `fs = 256 Hz` and `nperseg = 256`, frequency resolution is `fs / nperseg = 1 Hz`, which is sufficient to resolve the four EEG bands.

---

## Frequency bands

```python
FREQ_BANDS = {
    "delta": (1.0,  4.0),   # slow cortical activity, deep sleep
    "theta": (4.0,  8.0),   # working memory, frontal midline theta
    "alpha": (8.0, 13.0),   # resting-state rhythm; suppressed in AUD
    "beta":  (13.0, 30.0),  # active processing, motor cortex
}
```

**Alpha band is the primary marker of interest.** Chronic alcohol use disorder (AUD) is consistently associated with alpha-band suppression in published EEG literature. Theta and beta are included to give the classifier additional discriminative signal without hand-picking a single band.

---

## Functions

### `compute_band_power(signal, fs, fmin, fmax, nperseg) → float`

Estimates mean power spectral density (μV²/Hz) in the band `[fmin, fmax]` for a single electrode's time-series.

**Steps:**
1. Call `scipy.signal.welch(signal, fs=fs, nperseg=nperseg)`
2. Apply a boolean mask: `(freqs >= fmin) & (freqs <= fmax)`
3. Return `np.mean(psd[mask])`

Returns `0.0` if no frequency bin falls within the requested range (edge case guard).

---

### `compute_all_band_powers(signal, fs) → dict[str, float]`

Convenience wrapper. Calls `compute_band_power` for every entry in `FREQ_BANDS` and returns a dict:

```python
{
    "delta": 12.4,
    "theta": 8.1,
    "alpha": 5.3,
    "beta":  3.7,
}
```

---

## Usage

```python
from src.preprocessing import compute_band_power, compute_all_band_powers, FREQ_BANDS

signal = df[sample_cols].values[0]          # shape (256,)
alpha_power = compute_band_power(signal, fs=256, fmin=8.0, fmax=13.0)
all_bands   = compute_all_band_powers(signal, fs=256)
```
