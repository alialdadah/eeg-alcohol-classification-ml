"""
train.py
--------
Model training with subject-aware cross-validation.

Why GroupKFold?
~~~~~~~~~~~~~~~
The EEG data contains multiple trials from the same subject.  A naïve random
train/test split (as used in the original pipeline) will place different trials
from the same subject in both training and testing.  Because intra-subject EEG
is highly correlated, the model then learns subject identity rather than the
alcoholism biomarker — this is subject-level data leakage.

GroupKFold ensures every fold has a disjoint set of subjects in train and test.
With 16 subjects (8 per class) and n_splits=4, each test fold contains
approximately 4 subjects (2 alcoholic, 2 control).

Hyperparameter search
~~~~~~~~~~~~~~~~~~~~~
An inner GridSearchCV over the SVM regularisation parameter C is nested inside
the outer GroupKFold loop.  The scaler is fit on the training partition of each
fold (not the whole dataset) to prevent information leakage through the scaler.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ── Default hyperparameter grid ─────────────────────────────────────────────
DEFAULT_PARAM_GRID = {
    "svm__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "svm__kernel": ["rbf", "linear"],
}


def build_pipeline() -> Pipeline:
    """Return a standard scaler → SVM pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=42)),
        ]
    )


def cross_validate_subject_aware(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 4,
    param_grid: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run subject-aware cross-validation with inner hyperparameter search.

    Parameters
    ----------
    X : array, shape (n_trials, n_features)
    y : array, shape (n_trials,)
    groups : array, shape (n_trials,)
        Subject name per trial — passed to GroupKFold.
    n_splits : int
        Number of outer folds.  Must be ≤ number of unique subjects.
    param_grid : dict or None
        Grid for inner GridSearchCV.  Uses DEFAULT_PARAM_GRID if None.
    verbose : bool

    Returns
    -------
    dict with keys:
        fold_results  : list of per-fold metric dicts
        mean_*        : mean metric across folds
        std_*         : std metric across folds
        best_params   : most frequently selected hyperparams
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    outer_cv = GroupKFold(n_splits=n_splits)
    fold_results: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Inner CV for hyperparameter search (group-aware)
        # Need at least 2 classes and 2 groups in every inner fold.
        # With very small datasets fall back to training directly with default params.
        n_unique_groups_train = len(np.unique(groups_train))
        n_unique_classes_train = len(np.unique(y_train))
        inner_n_splits = min(3, n_unique_groups_train)

        pipeline = build_pipeline()
        used_grid_search = False
        best_params_used: dict = {}

        # Inner CV requires ≥2 classes and ≥2 groups in every inner training split.
        # With small datasets this may not hold; fall back to default params.
        can_do_inner_cv = n_unique_classes_train >= 2 and inner_n_splits >= 2
        if can_do_inner_cv:
            inner_cv = GroupKFold(n_splits=inner_n_splits)
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=inner_cv,
                scoring="f1",
                n_jobs=-1,
                refit=True,
                error_score=0.0,
            )
            try:
                grid_search.fit(X_train, y_train, groups=groups_train)
                best_model = grid_search.best_estimator_
                best_params_used = grid_search.best_params_
                used_grid_search = True
            except ValueError:
                # Fall back silently if inner CV still fails
                can_do_inner_cv = False

        if not can_do_inner_cv:
            pipeline.fit(X_train, y_train)
            best_model = pipeline

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "fold": fold_idx,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "test_subjects": list(np.unique(groups[test_idx])),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba)
                if len(np.unique(y_test)) > 1 else float("nan"),
            "best_params": best_params_used,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        fold_results.append(metrics)

        if verbose:
            print(
                f"  Fold {fold_idx}: "
                f"acc={metrics['accuracy']:.3f}  "
                f"f1={metrics['f1']:.3f}  "
                f"auc={metrics['roc_auc']:.3f}  "
                f"best={best_params_used}"
            )

    # Aggregate metrics
    def _mean(key: str) -> float:
        vals = [r[key] for r in fold_results if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(key: str) -> float:
        vals = [r[key] for r in fold_results if not np.isnan(r[key])]
        return float(np.std(vals)) if vals else float("nan")

    result = {
        "fold_results": fold_results,
        "mean_accuracy":  _mean("accuracy"),
        "std_accuracy":   _std("accuracy"),
        "mean_f1":        _mean("f1"),
        "std_f1":         _std("f1"),
        "mean_precision": _mean("precision"),
        "std_precision":  _std("precision"),
        "mean_recall":    _mean("recall"),
        "std_recall":     _std("recall"),
        "mean_roc_auc":   _mean("roc_auc"),
        "std_roc_auc":    _std("roc_auc"),
    }
    return result


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    best_params: dict | None = None,
) -> Pipeline:
    """
    Train a final pipeline on the full dataset using the best hyperparameters
    found during cross-validation.  Intended for persistence / deployment, not
    for reporting generalisation metrics.

    Parameters
    ----------
    X : array, shape (n_trials, n_features)
    y : array, shape (n_trials,)
    best_params : dict or None
        e.g. {'svm__C': 1.0, 'svm__kernel': 'rbf'}.
        If None, defaults are used.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    pipeline = build_pipeline()
    if best_params:
        pipeline.set_params(**best_params)
    pipeline.fit(X, y)
    return pipeline
