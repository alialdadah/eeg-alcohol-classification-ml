# `src/features.py`

## Purpose

Builds the feature matrix that gets passed to the ML model. Three modes are
available, each returning the same four outputs: `(X, y, groups, feature_names)`.

---

## The data layout problem (applies to all modes)

The raw CSV has one row per electrode per trial per subject. A single trial
produces 61 rows. Treating each row as an independent sample — as the original
code did — inflates sample count 61× and makes random splitting scatter the
same trial's electrode rows into both train and test.

Every function here groups on `(name, trial number)` first, so the output has
exactly **one row per trial**, not one row per electrode.

---

## Mode 1 — `build_feature_matrix` (band-power only)

**Default condition:** `S1 obj` (single-stimulus, no paradigm confound)

For each trial, computes Welch PSD band power for each electrode and flattens
into a feature vector:

```
[AF3_delta, AF3_theta, AF3_alpha, AF3_beta,
 AF4_delta, ...,
 TP8_beta]
```

**Shape:** `(n_trials, n_electrodes × 4)` — with all 61 electrodes: **244 features**

---

## Mode 2 — `build_p300_features` (P300 ERP only)

**Forced condition:** `S2 nomatch` (oddball paradigm required for P300)

For each trial, extracts P300 ERP features at parietal/central electrodes only:

```
[p300_CZ_mean_amp, p300_CZ_peak_amp, p300_CZ_peak_lat_ms, p300_CZ_auc,
 p300_PZ_mean_amp, ...,
 p300_POZ_auc]
```

**Shape:** `(n_trials, n_p300_electrodes × 4)` — default 5 electrodes: **20 features**

Low-dimensional relative to sample size — important for a small-n dataset.

---

## Mode 3 — `build_combined_features` (P300 + band-power)

**Forced condition:** `S2 nomatch`

Concatenates the P300 block and band-power block from the same trials:

```
[p300_CZ_mean_amp, ..., p300_POZ_auc,    ← P300 block (20 features)
 bp_AF3_delta,     ..., bp_TP8_beta]      ← band-power block (244 features)
```

**Shape:** `(n_trials, 264 features)`

Both blocks use the same set of trials in the same order, so they are properly
aligned in the feature matrix.

---

## Measured results (4-fold subject-held-out CV, n=16 subjects)

| Mode | Features | Accuracy | F1 | ROC-AUC | AUC ±SD |
|------|----------|----------|----|---------|---------|
| `band_power` (S1 obj) | 244 | 0.531 ± 0.211 | 0.529 | 0.527 | ±0.289 |
| `p300` (S2 nomatch) | 20 | 0.483 ± 0.144 | 0.559 | **0.587** | **±0.090** |
| `combined` (S2 nomatch) | 264 | 0.391 ± 0.237 | 0.427 | 0.584 | ±0.353 |

**Key observation**: P300-only has the **lowest fold-to-fold variance** by a large
margin (AUC ±0.090 vs ±0.289 and ±0.353). With only 20 features and 110–120
training samples per fold, the model generalises more consistently than the
high-dimensional modes. The combined mode suffers from the curse of dimensionality:
264 features with ~110 training samples causes the band-power block to add noise
rather than signal at this dataset size.

---

## Returns (all three functions)

| Name | Shape | Description |
|------|-------|-------------|
| `X` | `(n_trials, n_features)` | Feature matrix |
| `y` | `(n_trials,)` | `1` = alcoholic, `0` = control |
| `groups` | `(n_trials,)` | Subject name — passed to `GroupKFold` |
| `feature_names` | `list[str]` | One label per column of `X` |

---

## Electrode constants

```python
FRONTAL_ELECTRODES   # 26 frontal/prefrontal sites (AF*, F*, FC*, FP*, FT*)
# P300_ELECTRODES imported from preprocessing: ["CZ","PZ","P3","P4","POZ"]
```

---

## Usage

```python
from src.features import build_feature_matrix, build_p300_features, build_combined_features

# Band-power
X, y, groups, names = build_feature_matrix(df, condition="S1 obj")

# P300 only
X, y, groups, names = build_p300_features(df)

# Combined (recommended for S2 nomatch)
X, y, groups, names = build_combined_features(df)
```
