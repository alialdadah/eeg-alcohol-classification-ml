# `src/data_loader.py`

## Purpose

Loads the pre-formatted EEG CSV into a pandas DataFrame and validates its structure before any processing begins. Acts as the single entry point for raw data — nothing else in the pipeline reads files directly.

---

## Functions

### `load_eeg_data(path) → pd.DataFrame`

Reads `EEG_formatted.csv` (or the sample fixture) and runs five validation checks:

| Check | What it catches |
|-------|----------------|
| File exists | Gives a clear error with download instructions instead of a raw `FileNotFoundError` |
| Required columns present | Catches renamed or missing metadata columns |
| 256 sample columns present | Catches truncated or malformed exports |
| Subject identifiers are `'a'` or `'c'` | Catches encoding issues or wrong file |
| Drops unnamed index column | The CSV formatter adds an extra index column; this strips it silently |

Returns the validated DataFrame unchanged — no filtering or transformation happens here.

---

### `summarize_dataset(df) → None`

Prints a human-readable summary to stdout:

```
Dataset summary
  Total rows     : 28,548
  Electrodes     : 61
  Subjects (a)   : 8
  Subjects (c)   : 8
  Conditions     : ['S1 obj', 'S2 match', 'S2 nomatch']
  Unique trials  : 468
```

---

## Expected CSV schema

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Subject ID string, e.g. `co2a0000364` |
| `trial number` | int | Trial index within the session |
| `matching condition` | str | `S1 obj`, `S2 match`, or `S2 nomatch` |
| `sensor position` | str | Electrode label, e.g. `AF3`, `CZ` |
| `subject identifier` | str | `a` = alcoholic, `c` = control |
| `sample_0` … `sample_255` | float | EEG voltage (μV) at each time point |

---

## Constants

```python
REQUIRED_COLUMNS      # set of 5 metadata column names
EXPECTED_SAMPLE_COLS  # list of 'sample_0' … 'sample_255'
VALID_SUBJECT_IDS     # {'a', 'c'}
VALID_CONDITIONS      # {'S1 obj', 'S2 match', 'S2 nomatch'}
```

---

## Usage

```python
from src.data_loader import load_eeg_data, summarize_dataset

df = load_eeg_data("data/EEG_formatted.csv")
summarize_dataset(df)
```
