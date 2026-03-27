"""
data_loader.py
--------------
Loads and validates the pre-formatted EEG CSV.

The expected format (produced by the data-formatting notebook) is:
    name, trial number, matching condition, sensor position,
    subject identifier, sample_0 … sample_255

Each row is one electrode recording for one trial for one subject.
"""

import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = {
    "name",
    "trial number",
    "matching condition",
    "sensor position",
    "subject identifier",
}

EXPECTED_SAMPLE_COLS = [f"sample_{i}" for i in range(256)]

VALID_SUBJECT_IDS = {"a", "c"}
VALID_CONDITIONS = {"S1 obj", "S2 match", "S2 nomatch"}


def load_eeg_data(path: str | Path) -> pd.DataFrame:
    """
    Load the pre-formatted EEG CSV and run basic validation.

    Parameters
    ----------
    path : str or Path
        Path to EEG_formatted.csv (or the sample file for testing).

    Returns
    -------
    pd.DataFrame
        Validated DataFrame ready for feature extraction.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns or expected subject IDs are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Download instructions: see data/README.md"
        )

    df = pd.read_csv(path)

    # Drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    missing_samples = [c for c in EXPECTED_SAMPLE_COLS if c not in df.columns]
    if missing_samples:
        raise ValueError(
            f"CSV is missing {len(missing_samples)} sample columns "
            f"(e.g. {missing_samples[:3]})"
        )

    found_ids = set(df["subject identifier"].unique())
    if not found_ids.issubset(VALID_SUBJECT_IDS):
        unexpected = found_ids - VALID_SUBJECT_IDS
        raise ValueError(f"Unexpected subject identifiers: {unexpected}")

    return df


def summarize_dataset(df: pd.DataFrame) -> None:
    """Print a concise summary of the loaded dataset."""
    n_subjects_a = df[df["subject identifier"] == "a"]["name"].nunique()
    n_subjects_c = df[df["subject identifier"] == "c"]["name"].nunique()
    print(f"Dataset summary")
    print(f"  Total rows     : {len(df):,}")
    print(f"  Electrodes     : {df['sensor position'].nunique()}")
    print(f"  Subjects (a)   : {n_subjects_a}")
    print(f"  Subjects (c)   : {n_subjects_c}")
    print(f"  Conditions     : {sorted(df['matching condition'].unique())}")
    trials = df.groupby(["name", "trial number", "matching condition"]).ngroups
    print(f"  Unique trials  : {trials}")
