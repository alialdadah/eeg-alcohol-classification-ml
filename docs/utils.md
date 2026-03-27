# `src/utils.py`

## Purpose

Small utility functions that don't belong to any specific pipeline stage: random seed management, result serialisation, and sample data generation.

---

## Functions

### `set_seed(seed=42) → None`

Fixes `random.seed` and `numpy.random.seed` for reproducibility. Called once at the start of `main.py` before any data processing. Does not fix scikit-learn's internal state directly — that is handled by passing `random_state=42` to `SVC`.

---

### `save_results(results, path) → None`

Serialises the CV results dict to a JSON file. The dict contains numpy arrays (`y_test`, `y_pred`, `y_proba`) that are not JSON-serialisable by default; a recursive `_convert` helper converts them to plain Python lists before writing.

Output example (`reports/cv_results.json`):
```json
{
  "mean_accuracy": 0.531,
  "std_accuracy": 0.211,
  "mean_f1": 0.529,
  "fold_results": [
    {
      "fold": 1,
      "accuracy": 0.775,
      "y_test": [1, 0, 1, ...],
      ...
    }
  ]
}
```

---

### `load_results(path) → dict`

Reads a previously saved `cv_results.json` back into a Python dict. Note: arrays are loaded as plain lists; convert back to numpy as needed before passing to `evaluate.py` functions.

---

### `make_sample_csv(source_path, output_path, n_subjects_per_class, n_trials_per_subject, condition, seed) → None`

Creates a small reproducible sample CSV for unit tests or demos. Selects `n_subjects_per_class` subjects per class and `n_trials_per_subject` trials from each, keeping all 61 electrodes. Used to generate `data/sample/eeg_sample.csv`.

---

## Usage

```python
from src.utils import set_seed, save_results, load_results

set_seed(42)
# ... run pipeline ...
save_results(cv_results, "reports/cv_results.json")

# Later, in a notebook or separate script:
cv = load_results("reports/cv_results.json")
```
