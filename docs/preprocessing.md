# `src/preprocessing.py`

## Purpose

All signal processing happens here. Two families of features are supported:
band-power (frequency domain) and P300 ERP (time domain).

---

## Family 1 — Band-power features

### Why Welch instead of raw FFT

The original project used raw FFT magnitude on 1-second epochs. Raw FFT on short
epochs has high spectral variance — the estimate is noisy and changes significantly
between trials from the same subject. Welch's method averages overlapping
periodogram segments, reducing that variance substantially. The trade-off (slightly
lower frequency resolution) is acceptable here because 1 Hz resolution is
sufficient to separate the four EEG bands.

### Frequency bands

```python
FREQ_BANDS = {
    "delta": (1.0,  4.0),   # slow cortical activity
    "theta": (4.0,  8.0),   # working memory, frontal midline theta
    "alpha": (8.0, 13.0),   # resting rhythm — suppressed in AUD
    "beta":  (13.0, 30.0),  # active processing, arousal
}
```

Alpha (8–13 Hz) is the primary band of interest. Chronic alcohol use disorder is
consistently associated with alpha-band suppression in the EEG literature.

### `compute_band_power(signal, fs, fmin, fmax, nperseg) → float`

Uses `scipy.signal.welch`. Returns mean PSD (μV²/Hz) in the band.

### `compute_all_band_powers(signal, fs) → dict[str, float]`

Convenience wrapper — returns a dict for all four bands at once.

---

## Family 2 — P300 ERP features

### What is P300

The P300 is a **time-domain** event-related potential: a positive voltage deflection
peaking approximately 300 ms after an unexpected (oddball) stimulus. It is not
visible in the frequency spectrum — it must be extracted directly from the
time-series within a post-stimulus window.

P300 amplitude is **one of the most replicated neurophysiological biomarkers of
alcohol use disorder**. The SMNI CMI dataset was originally collected by Henri
Begleiter's lab specifically to study P300 in AUD families.

### Electrode locations

```python
P300_ELECTRODES = ["CZ", "PZ", "P3", "P4", "POZ"]
```

These parietal and central-midline sites are where P300 is maximal. The original
project code incorrectly removed all of them — this implementation restores them.

### Time window

```python
P300_WINDOW_MS = (250, 500)  # milliseconds post-stimulus
# At 256 Hz: samples 64 to 128
```

No pre-stimulus baseline correction is applied because the formatted CSV epochs
start at stimulus onset (t = 0) with no pre-stimulus samples. Using the raw
window amplitude is standard practice in this situation.

### `extract_p300_features(signal, fs, window_ms) → dict`

Extracts four features from the P300 window of a single electrode:

| Feature | Description |
|---------|-------------|
| `mean_amp` | Mean amplitude in 250–500 ms (μV) |
| `peak_amp` | Maximum amplitude in 250–500 ms (μV) |
| `peak_lat_ms` | Latency of the peak in milliseconds |
| `auc` | Area under the waveform (trapezoidal rule) |

---

## Constants

```python
FREQ_BANDS          # dict: band name → (fmin, fmax)
P300_WINDOW_MS      # (250, 500)
P300_ELECTRODES     # ["CZ", "PZ", "P3", "P4", "POZ"]
P300_FEATURE_NAMES  # ["mean_amp", "peak_amp", "peak_lat_ms", "auc"]
```
