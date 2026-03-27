# `src/train.py`

## Purpose

Trains and evaluates the SVM classifier using a nested, subject-aware cross-validation scheme that prevents data leakage.

---

## Why GroupKFold instead of `train_test_split`

This is the most important design decision in the pipeline.

The dataset has **16 subjects** (8 alcoholic, 8 control), each contributing ~10 trials per condition. A single subject's trials are highly correlated — they come from the same brain recorded in the same session.

If you split rows randomly:
- Subject X has 10 trials → 7 go to train, 3 go to test
- The model sees subject X during training and is tested on subject X
- It learns to recognise subject X's EEG signature (which is very stable)
- Reported accuracy measures **person recognition**, not **alcoholism classification**

`GroupKFold(groups=subject_name)` guarantees that every test fold contains **subjects not present in training**. This is the only valid evaluation for small neuroimaging datasets.

---

## Nested cross-validation structure

```
Outer loop: GroupKFold(n_splits=4)
│  Each fold: 12 subjects train, 4 subjects test
│
└─ Inner loop: GroupKFold(n_splits=3)  [on training subjects only]
       Grid search over:
         svm__C      ∈ {0.01, 0.1, 1, 10, 100}
         svm__kernel ∈ {'rbf', 'linear'}
       Scored by F1 on the inner validation subjects
```

The inner loop finds the best hyperparameters without touching the outer test subjects. The `StandardScaler` is fitted inside the pipeline — it sees only the training fold's data, never the test fold's statistics.

---

## Pipeline

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(probability=True, random_state=42)),
])
```

`probability=True` is required for `predict_proba`, which is needed for ROC-AUC computation.

---

## Functions

### `build_pipeline() → Pipeline`
Returns a fresh, unfitted scaler+SVM pipeline.

### `cross_validate_subject_aware(X, y, groups, n_splits, param_grid, verbose) → dict`

Runs the full nested CV and returns a results dict containing:

| Key | Description |
|-----|-------------|
| `fold_results` | List of per-fold dicts with `y_test`, `y_pred`, `y_proba`, metrics, best params |
| `mean_accuracy` / `std_accuracy` | Across-fold mean and standard deviation |
| `mean_f1` / `std_f1` | Same for F1 |
| `mean_precision` / `std_precision` | Same for precision |
| `mean_recall` / `std_recall` | Same for recall |
| `mean_roc_auc` / `std_roc_auc` | Same for ROC-AUC |

### `train_final_model(X, y, best_params) → Pipeline`
Trains on the full dataset with specified params. For saving/deployment only — never use its predictions for reporting generalisation performance.

---

## Edge case: small datasets

When running on the 4-subject sample fixture, the inner CV may receive a training fold with only 1 class (e.g. 2 subjects → 1 trial each → after inner split, one fold has 1 class). The code detects this and falls back to fitting the pipeline directly with default parameters rather than crashing.

---

## Usage

```python
from src.train import cross_validate_subject_aware

cv_results = cross_validate_subject_aware(
    X, y, groups,
    n_splits=4,
    verbose=True,
)
print(cv_results["mean_accuracy"], cv_results["std_accuracy"])
```
