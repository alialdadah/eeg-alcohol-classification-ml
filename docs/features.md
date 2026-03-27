# `src/features.py`

## Purpose

Builds the feature matrix that gets passed to the machine learning model. This module is responsible for converting the raw electrode×row DataFrame into one feature vector per EEG trial, with correct labels and subject group IDs for cross-validation.

---

## The data layout problem

The raw CSV has one row per electrode per trial per subject. A single 1-second EEG trial from one subject produces 61 rows (one per electrode). If you ignore this and treat each row as an independent sample — as the original code did — you get 61× more "samples" than you actually have, and random train/test splitting will scatter rows from the same trial into both sets.

This module solves that by grouping on `(name, trial number)` before extracting any features, so the output has exactly one row per trial.

---

## Feature construction

For each unique `(subject_name, trial_number, condition)` group:

```
for each electrode in electrode_list:
    signal = 256-sample time series for that electrode
    for each band in [delta, theta, alpha, beta]:
        power = Welch PSD mean in that band's frequency range
        → append to feature vector
```

**Output shape:** `(n_trials, n_electrodes × 4)`

With all 61 electrodes: **244 features per trial**.
With `--frontal-only` (26 electrodes): **104 features per trial**.

If an electrode is missing from a trial (data quality gap), all 4 of its band values are filled with `0.0`.

---

## Returns

`build_feature_matrix` returns four objects:

| Name | Shape | Description |
|------|-------|-------------|
| `X` | `(n_trials, n_features)` | Feature matrix |
| `y` | `(n_trials,)` | Labels: `1` = alcoholic, `0` = control |
| `groups` | `(n_trials,)` | Subject name string per trial — used by GroupKFold |
| `feature_names` | `list[str]` | e.g. `['AF3_delta', 'AF3_theta', ..., 'TP8_beta']` |

The `groups` array is critical. It is passed to `GroupKFold.split()` in `train.py` so that every fold keeps all trials from one subject together. Without it, subject-level leakage occurs.

---

## Electrode sets

```python
FRONTAL_ELECTRODES  # 26 electrodes: AF*, F*, FC*, FP*, FT*
# Default: all 61 electrodes present in the data
```

The frontal set was the original project's region of interest (alpha asymmetry).
Using all electrodes lets the SVM discover which regions are most discriminative
without imposing a prior.

---

## Condition choice

The recommended condition is `'S1 obj'` (default). This is a **single-stimulus paradigm** — one image is shown, no match/nomatch judgment required. Using it avoids confounding the alcoholism group difference with cognitive processing differences that arise from the match/nomatch task.

`'S2 nomatch'` (the oddball condition that elicits P300) can be selected with `--condition "S2 nomatch"` but requires care in interpretation.

---

## Usage

```python
from src.features import build_feature_matrix

X, y, groups, feature_names = build_feature_matrix(
    df,
    condition="S1 obj",
    electrodes=None,    # None = all 61
    verbose=True,
)
```
