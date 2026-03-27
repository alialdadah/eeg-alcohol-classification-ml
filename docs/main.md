# `main.py`

## Purpose

Command-line entry point for the full pipeline. Wires together all `src/` modules in the correct order and exposes every configurable parameter as a CLI argument.

---

## Pipeline order

```
1. parse_args()          ← read CLI flags
2. set_seed()            ← fix numpy/random for reproducibility
3. load_eeg_data()       ← load + validate CSV
4. summarize_dataset()   ← print data overview
5. build_feature_matrix()← compute Welch PSD features per trial
6. cross_validate_subject_aware() ← nested GroupKFold + GridSearchCV
7. print_cv_summary()    ← console metrics table
8. save_results()        ← write cv_results.json
9. plot_*()              ← save PNG figures (unless --no-plots)
10. print honest summary ← final readable output block
```

---

## CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/EEG_formatted.csv` | Path to formatted EEG CSV |
| `--condition` | `S1 obj` | EEG paradigm condition to classify |
| `--n-folds` | `4` | Outer GroupKFold folds |
| `--frontal-only` | off | Restrict to 26 frontal electrodes |
| `--output-dir` | `reports` | Where to save plots and JSON |
| `--seed` | `42` | Random seed |
| `--no-plots` | off | Skip plot generation |

---

## Examples

```bash
# Full run on complete dataset
python main.py --data data/EEG_formatted.csv

# Smoke test with sample data
python main.py --data data/sample/eeg_sample.csv --n-folds 2

# Frontal electrodes only, S2 nomatch condition
python main.py --data data/EEG_formatted.csv --condition "S2 nomatch" --frontal-only

# Suppress plots (e.g. on a headless server)
python main.py --data data/EEG_formatted.csv --no-plots
```

---

## Automatic n_folds reduction

If the loaded data has fewer subjects than `n_folds × 2`, the script automatically reduces `n_folds` and prints a warning. This prevents confusing GroupKFold errors when running on the sample fixture.

---

## Output files

All outputs go to `--output-dir` (default `reports/`):

| File | Description |
|------|-------------|
| `cv_results.json` | Full per-fold metrics and predictions |
| `confusion_matrix.png` | Aggregated CM across all folds |
| `roc_curve.png` | Per-fold ROC curves + macro average |
| `fold_metrics.png` | Bar chart of accuracy/F1/AUC per fold |
