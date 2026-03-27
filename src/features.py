"""
features.py
-----------
Build feature matrices from the loaded EEG DataFrame.

Three feature modes
~~~~~~~~~~~~~~~~~~~
  band_power  — Welch PSD band power per electrode (delta/theta/alpha/beta).
                Recommended condition: S1 obj (clean single-stimulus baseline).
                Shape per trial: n_electrodes × 4.

  p300        — P300 ERP features per parietal/central electrode.
                Requires condition: S2 nomatch (oddball paradigm elicits P300).
                Features per electrode: mean_amp, peak_amp, peak_lat_ms, auc.
                Shape per trial: n_p300_electrodes × 4.

  combined    — P300 features at parietal electrodes  +  band-power features
                at all electrodes, both from S2 nomatch trials.
                Shape per trial: (n_p300_electrodes × 4) + (n_bp_electrodes × 4).

Subject leakage note
~~~~~~~~~~~~~~~~~~~~
All three functions return (X, y, groups, feature_names).  The groups array
(subject name per trial) must be passed to GroupKFold in train.py so that
no subject appears in both training and test within a fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocessing import (
    compute_band_power,
    extract_p300_features,
    FREQ_BANDS,
    P300_ELECTRODES,
    P300_FEATURE_NAMES,
)


# ── Electrode sets ─────────────────────────────────────────────────────────

FRONTAL_ELECTRODES: list[str] = [
    "AF3", "AF4", "AF7", "AF8", "AFZ",
    "F1",  "F2",  "F3",  "F4",  "F5",  "F6",  "F7",  "F8",  "FZ",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCZ",
    "FP1", "FP2", "FPZ",
    "FT7", "FT8",
]

SAMPLE_COLUMNS: list[str] = [f"sample_{i}" for i in range(256)]


# ── Shared helpers ─────────────────────────────────────────────────────────

def _iter_trials(df: pd.DataFrame, condition: str, verbose: bool, desc: str):
    """Filter to condition, group by trial, return an optionally tqdm-wrapped iterator."""
    subset = df[df["matching condition"] == condition].copy()
    if len(subset) == 0:
        raise ValueError(
            f"No rows found for condition '{condition}'. "
            f"Available: {df['matching condition'].unique().tolist()}"
        )
    grouped = subset.groupby(["name", "trial number", "subject identifier"], sort=False)
    return tqdm(grouped, desc=desc, unit="trial") if verbose else grouped


def _get_signal(trial_df: pd.DataFrame, electrode: str) -> np.ndarray | None:
    """Return the 256-sample signal for one electrode in a trial, or None if missing."""
    rows = trial_df[trial_df["sensor position"] == electrode]
    if len(rows) == 0:
        return None
    return rows[SAMPLE_COLUMNS].values[0].astype(float)


# ── Feature builder 1: band-power only ────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    condition: str = "S1 obj",
    fs: int = 256,
    electrodes: list[str] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Band-power feature matrix (delta / theta / alpha / beta per electrode).

    Parameters
    ----------
    df : pd.DataFrame
    condition : str
        'S1 obj' recommended — single-stimulus, no paradigm confound.
    fs : int
    electrodes : list[str] or None
        Electrodes to include.  None = all 61 present in the data.
    verbose : bool

    Returns
    -------
    X            : (n_trials, n_electrodes * 4)
    y            : (n_trials,)  1=alcoholic, 0=control
    groups       : (n_trials,)  subject name
    feature_names: list[str]
    """
    subset = df[df["matching condition"] == condition]
    if len(subset) == 0:
        raise ValueError(
            f"No rows found for condition '{condition}'. "
            f"Available: {df['matching condition'].unique().tolist()}"
        )
    if electrodes is None:
        electrodes = sorted(subset["sensor position"].unique())

    feature_names = [f"{e}_{b}" for e in electrodes for b in FREQ_BANDS]
    n_features = len(feature_names)

    X_rows, y_list, groups_list = [], [], []

    iterator = _iter_trials(df, condition, verbose, desc="Band-power features")

    for (subject_name, trial_num, subject_id), trial_df in iterator:
        row: list[float] = []
        for elec in electrodes:
            sig = _get_signal(trial_df, elec)
            if sig is None:
                row.extend([0.0] * len(FREQ_BANDS))
                continue
            for fmin, fmax in FREQ_BANDS.values():
                row.append(compute_band_power(sig, fs=fs, fmin=fmin, fmax=fmax))

        assert len(row) == n_features
        X_rows.append(row)
        y_list.append(1 if subject_id == "a" else 0)
        groups_list.append(subject_name)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_list, dtype=int)
    groups = np.array(groups_list, dtype=str)

    if verbose:
        print(f"\nBand-power matrix: {X.shape[0]} trials × {X.shape[1]} features  "
              f"(alcoholic={y.sum()}, control={(y==0).sum()})")
    return X, y, groups, feature_names


# ── Feature builder 2: P300 only ──────────────────────────────────────────

def build_p300_features(
    df: pd.DataFrame,
    fs: int = 256,
    electrodes: list[str] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    P300 ERP feature matrix from S2 nomatch (oddball) trials.

    For each trial, extracts mean amplitude, peak amplitude, peak latency,
    and area-under-curve in the 250–500 ms window at parietal / central
    electrodes where P300 is maximal.

    Parameters
    ----------
    df : pd.DataFrame
    fs : int
    electrodes : list[str] or None
        P300 electrode set.  None = P300_ELECTRODES (CZ, PZ, P3, P4, POZ).
    verbose : bool

    Returns
    -------
    X            : (n_trials, n_electrodes * 4)
    y            : (n_trials,)
    groups       : (n_trials,)
    feature_names: list[str]   e.g. 'p300_PZ_peak_amp'
    """
    if electrodes is None:
        electrodes = P300_ELECTRODES

    feature_names = [
        f"p300_{e}_{feat}" for e in electrodes for feat in P300_FEATURE_NAMES
    ]
    n_features = len(feature_names)

    X_rows, y_list, groups_list = [], [], []

    iterator = _iter_trials(df, "S2 nomatch", verbose, desc="P300 features")

    for (subject_name, trial_num, subject_id), trial_df in iterator:
        row: list[float] = []
        for elec in electrodes:
            sig = _get_signal(trial_df, elec)
            if sig is None:
                row.extend([0.0] * len(P300_FEATURE_NAMES))
                continue
            feats = extract_p300_features(sig, fs=fs)
            row.extend([feats[k] for k in P300_FEATURE_NAMES])

        assert len(row) == n_features
        X_rows.append(row)
        y_list.append(1 if subject_id == "a" else 0)
        groups_list.append(subject_name)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_list, dtype=int)
    groups = np.array(groups_list, dtype=str)

    if verbose:
        print(f"\nP300 matrix: {X.shape[0]} trials × {X.shape[1]} features  "
              f"(alcoholic={y.sum()}, control={(y==0).sum()})")
    return X, y, groups, feature_names


# ── Feature builder 3: combined P300 + band-power ─────────────────────────

def build_combined_features(
    df: pd.DataFrame,
    fs: int = 256,
    bp_electrodes: list[str] | None = None,
    p300_electrodes: list[str] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Combined P300 ERP + band-power feature matrix from S2 nomatch trials.

    Both feature families are extracted from the same set of trials, so the
    feature vectors are properly aligned.  Using S2 nomatch means the P300
    component is present in the signal, and the band-power features capture
    spectral differences on the same epochs.

    Feature layout per trial
    ------------------------
      [p300_CZ_mean_amp, p300_CZ_peak_amp, ..., p300_POZ_auc,   ← P300 block
       bp_AF3_delta,     bp_AF3_theta,     ..., bp_TP8_beta]     ← band-power block

    Parameters
    ----------
    df : pd.DataFrame
    fs : int
    bp_electrodes : list[str] or None
        Electrodes for band-power block.  None = all 61.
    p300_electrodes : list[str] or None
        Electrodes for P300 block.  None = P300_ELECTRODES (5 parietal/central).
    verbose : bool

    Returns
    -------
    X            : (n_trials, n_p300_feats + n_bp_feats)
    y            : (n_trials,)
    groups       : (n_trials,)
    feature_names: list[str]
    """
    condition = "S2 nomatch"

    subset = df[df["matching condition"] == condition]
    if len(subset) == 0:
        raise ValueError(
            "Combined features require 'S2 nomatch' condition — "
            f"not found. Available: {df['matching condition'].unique().tolist()}"
        )

    if p300_electrodes is None:
        p300_electrodes = P300_ELECTRODES
    if bp_electrodes is None:
        bp_electrodes = sorted(subset["sensor position"].unique())

    p300_names = [f"p300_{e}_{f}" for e in p300_electrodes for f in P300_FEATURE_NAMES]
    bp_names   = [f"bp_{e}_{b}"   for e in bp_electrodes   for b in FREQ_BANDS]
    feature_names = p300_names + bp_names
    n_features = len(feature_names)

    X_rows, y_list, groups_list = [], [], []

    iterator = _iter_trials(df, condition, verbose, desc="Combined features")

    for (subject_name, trial_num, subject_id), trial_df in iterator:
        row: list[float] = []

        # ── P300 block ───────────────────────────────────────────────────
        for elec in p300_electrodes:
            sig = _get_signal(trial_df, elec)
            if sig is None:
                row.extend([0.0] * len(P300_FEATURE_NAMES))
                continue
            feats = extract_p300_features(sig, fs=fs)
            row.extend([feats[k] for k in P300_FEATURE_NAMES])

        # ── Band-power block ─────────────────────────────────────────────
        for elec in bp_electrodes:
            sig = _get_signal(trial_df, elec)
            if sig is None:
                row.extend([0.0] * len(FREQ_BANDS))
                continue
            for fmin, fmax in FREQ_BANDS.values():
                row.append(compute_band_power(sig, fs=fs, fmin=fmin, fmax=fmax))

        assert len(row) == n_features, (
            f"Feature length mismatch: expected {n_features}, got {len(row)}"
        )

        X_rows.append(row)
        y_list.append(1 if subject_id == "a" else 0)
        groups_list.append(subject_name)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_list, dtype=int)
    groups = np.array(groups_list, dtype=str)

    if verbose:
        n_a = y.sum()
        n_c = len(y) - n_a
        p300_dim = len(p300_names)
        bp_dim   = len(bp_names)
        print(
            f"\nCombined matrix : {X.shape[0]} trials × {X.shape[1]} features  "
            f"(alcoholic={n_a}, control={n_c})\n"
            f"  P300 block     : {p300_dim} features  "
            f"({len(p300_electrodes)} electrodes × {len(P300_FEATURE_NAMES)} ERP features)\n"
            f"  Band-power block: {bp_dim} features  "
            f"({len(bp_electrodes)} electrodes × {len(FREQ_BANDS)} bands)"
        )

    return X, y, groups, feature_names
