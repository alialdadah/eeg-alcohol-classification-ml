"""
utils.py
--------
Miscellaneous utility functions: result serialisation, reproducibility,
and a helper for creating a small sample dataset.
"""

from __future__ import annotations

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: dict, path: str | Path) -> None:
    """
    Serialise CV results to a JSON file.

    Non-serialisable values (numpy arrays) are converted to plain lists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    serialisable = _convert(results)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str | Path) -> dict:
    """Load CV results from a JSON file."""
    with open(path) as f:
        return json.load(f)


def make_sample_csv(
    source_path: str | Path,
    output_path: str | Path,
    n_subjects_per_class: int = 2,
    n_trials_per_subject: int = 2,
    condition: str = "S1 obj",
    seed: int = 42,
) -> None:
    """
    Create a small reproducible sample CSV for unit tests and demos.

    Parameters
    ----------
    source_path : str or Path
        Path to the full EEG_formatted.csv.
    output_path : str or Path
        Where to write the sample CSV.
    n_subjects_per_class : int
        How many subjects to include per class.
    n_trials_per_subject : int
        How many trials to take per subject.
    condition : str
        EEG condition to subset.
    seed : int
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(source_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    samples = []
    for sid in ["a", "c"]:
        names = df[df["subject identifier"] == sid]["name"].unique()
        chosen_names = rng.choice(names, size=min(n_subjects_per_class, len(names)), replace=False)
        for name in chosen_names:
            sub_df = df[(df["name"] == name) & (df["matching condition"] == condition)]
            trials = sub_df["trial number"].unique()
            chosen_trials = rng.choice(
                trials, size=min(n_trials_per_subject, len(trials)), replace=False
            )
            samples.append(sub_df[sub_df["trial number"].isin(chosen_trials)])

    sample_df = pd.concat(samples, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)
    print(
        f"Sample CSV written to {output_path} "
        f"({len(sample_df)} rows, {sample_df['name'].nunique()} subjects)"
    )
