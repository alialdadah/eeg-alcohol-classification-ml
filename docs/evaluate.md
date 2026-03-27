# `src/evaluate.py`

## Purpose

Generates metric reports and visualisations from the cross-validation results dict returned by `train.cross_validate_subject_aware`. All plots are saved to disk (non-interactive backend) so the pipeline runs cleanly in environments without a display.

---

## Functions

### `print_cv_summary(cv_results) → None`

Prints a formatted table of per-fold and aggregate metrics to stdout:

```
============================================================
  Subject-Aware Cross-Validation Results
============================================================
  Fold     Acc      F1    Prec     Rec     AUC  Test subjects
  -----------------------------------------------------------
     1   0.775   0.791   0.739   0.850   0.878  0369, 0375, 0340, 0345
     2   0.375   0.359   0.368   0.350   0.335  0368, 0372, 0339, 0344
     3   0.275   0.216   0.235   0.200   0.165  0365, 0371, 0338, 0342
     4   0.700   0.750   0.643   0.900   0.732  0364, 0370, 0337, 0341
  -----------------------------------------------------------
  Mean   0.531   0.529   0.496   0.575   0.527
   ±SD   0.211   0.247   0.203   0.305   0.289
============================================================
```

Also prints an interpretation reminder about high fold-to-fold variance.

---

### `plot_confusion_matrix(cv_results, save_path) → None`

Aggregates `y_test` and `y_pred` across all folds and plots a single confusion matrix.

- Labels: `["Control", "Alcoholic"]`
- Saved as PNG if `save_path` is provided; shown interactively otherwise.

---

### `plot_roc_curve(cv_results, save_path) → None`

Plots one ROC curve per fold (thin, semi-transparent) plus the macro-average curve (thick, navy). Interpolates all fold curves to a common FPR grid for averaging.

Shows the spread in performance across folds visually — important for communicating uncertainty when n_subjects is small.

---

### `plot_fold_metrics(cv_results, save_path) → None`

Grouped bar chart showing Accuracy, F1, and ROC-AUC for each fold side by side, with a dashed 0.50 chance baseline. Makes it easy to see which folds drove the mean up or down.

---

## Backend note

```python
matplotlib.use("Agg")
```

This is set at import time so the module works correctly in scripts, servers, and environments without a display. It means `plt.show()` does nothing — all output must go through `save_path`.

---

## Usage

```python
from src.evaluate import print_cv_summary, plot_confusion_matrix, plot_roc_curve, plot_fold_metrics

print_cv_summary(cv_results)
plot_confusion_matrix(cv_results, save_path="reports/confusion_matrix.png")
plot_roc_curve(cv_results,        save_path="reports/roc_curve.png")
plot_fold_metrics(cv_results,     save_path="reports/fold_metrics.png")
```
