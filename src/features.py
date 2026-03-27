"""
features.py
-----------
Build the feature matrix from the loaded EEG DataFrame.

Feature construction
~~~~~~~~~~~~~~~~~~~~
For each EEG trial (identified by subject × trial_number × condition):

  1. Retrieve the time-series for every electrode present in that trial.
  2. Compute Welch PSD band power for each of four bands
     (delta, theta, alpha, beta).
  3. Concatenate per-electrode band powers into a flat feature vector.

Result shape:  (n_trials, n_electrodes × 4)

With all 61 electrodes this yields 244 features.  Missing electrodes are
filled with 0.

Subject leakage note
~~~~~~~~~~~~~~~~~~~~
Labels and group IDs are returned separately.  The group array (subject names)
must be passed to GroupKFold or LeaveOneGroupOut in train.py so that every
fold keeps all trials from one subject together.  This prevents subject-level
data leakage — the main flaw in the original pipeline.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocessing import compute_band_power, FREQ_BANDS


# ── Default electrode sets ──────────────────────────────────────────────────
# All 61 electrodes are used by default.  Frontal-only and full sets are
# provided for optional experimentation.

FRONTAL_ELECTRODES = [
    "AF3", "AF4", "AF7", "AF8", "AFZ",
    "F1",  "F2",  "F3",  "F4",  "F5",  "F6",  "F7",  "F8",  "FZ",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCZ",
    "FP1", "FP2", "FPZ",
    "FT7", "FT8",
]

SAMPLE_COLUMNS = [f"sample_{i}" for i in range(256)]


def build_feature_matrix(
    df: pd.DataFrame,
    condition: str = "S1 obj",
    fs: int = 256,
    electrodes: list[str] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build a feature matrix from the EEG DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_eeg_data().
    condition : str
        EEG paradigm condition to use.  'S1 obj' is recommended because it is
        a single-stimulus condition and avoids confounding alcoholism with the
        match/nomatch P300 paradigm.
    fs : int
        Sampling frequency in Hz.
    electrodes : list[str] or None
        Electrode names to include.  None = all electrodes present in the data.
    verbose : bool
        Show a tqdm progress bar.

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_electrodes * n_bands)
        Feature matrix.
    y : np.ndarray, shape (n_trials,)
        Binary labels: 1 = alcoholic ('a'), 0 = control ('c').
    groups : np.ndarray, shape (n_trials,)
        Subject name for each trial (used with GroupKFold to prevent leakage).
    feature_names : list[str]
        Human-readable name for each column of X.
    """
    subset = df[df["matching condition"] == condition].copy()

    if len(subset) == 0:
        raise ValueError(
            f"No rows found for condition '{condition}'. "
            f"Available: {df['matching condition'].unique().tolist()}"
        )

    if electrodes is None:
        electrodes = sorted(subset["sensor position"].unique())

    # Pre-build feature name list
    feature_names: list[str] = [
        f"{elec}_{band}"
        for elec in electrodes
        for band in FREQ_BANDS
    ]
    n_features = len(feature_names)

    X_rows: list[list[float]] = []
    y_list: list[int] = []
    groups_list: list[str] = []

    grouped = subset.groupby(
        ["name", "trial number", "subject identifier"],
        sort=False,
    )

    iterator = tqdm(grouped, desc="Extracting features", unit="trial") if verbose else grouped

    for (subject_name, trial_num, subject_id), trial_df in iterator:
        row: list[float] = []

        for elec in electrodes:
            elec_rows = trial_df[trial_df["sensor position"] == elec]

            if len(elec_rows) == 0:
                # Missing electrode: fill all bands with 0
                row.extend([0.0] * len(FREQ_BANDS))
                continue

            signal = elec_rows[SAMPLE_COLUMNS].values[0].astype(float)

            for fmin, fmax in FREQ_BANDS.values():
                power = compute_band_power(signal, fs=fs, fmin=fmin, fmax=fmax)
                row.append(power)

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
        print(
            f"\nFeature matrix: {X.shape[0]} trials × {X.shape[1]} features  "
            f"(alcoholic={n_a}, control={n_c})"
        )

    return X, y, groups, feature_names
