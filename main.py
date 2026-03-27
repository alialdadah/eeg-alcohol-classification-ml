"""
main.py
-------
Entry point for the EEG Alcohol Classification pipeline.

Usage
-----
    python main.py --data data/EEG_formatted.csv
    python main.py --data data/sample/eeg_sample.csv --n-folds 2
    python main.py --data data/EEG_formatted.csv --condition "S2 nomatch" --n-folds 4

The script will:
  1. Load and validate the EEG data.
  2. Build the feature matrix (band power per electrode per trial).
  3. Run subject-aware cross-validation with inner hyperparameter search.
  4. Print metrics and save plots + results JSON to reports/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_loader import load_eeg_data, summarize_dataset
from src.features import build_feature_matrix
from src.train import cross_validate_subject_aware
from src.evaluate import (
    print_cv_summary,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_fold_metrics,
)
from src.utils import set_seed, save_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEG alcohol classification — subject-aware SVM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/EEG_formatted.csv",
        help="Path to EEG_formatted.csv (or the sample CSV for testing)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="S1 obj",
        choices=["S1 obj", "S2 match", "S2 nomatch"],
        help=(
            "EEG paradigm condition to use for classification. "
            "'S1 obj' is the single-stimulus condition (recommended). "
            "'S2 nomatch' corresponds to the P300 oddball paradigm."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help=(
            "Number of outer GroupKFold folds. "
            "Each fold holds out a disjoint set of subjects for testing. "
            "Must be ≤ min(n_alcoholic_subjects, n_control_subjects)."
        ),
    )
    parser.add_argument(
        "--frontal-only",
        action="store_true",
        help=(
            "Restrict features to frontal electrodes only "
            "(AF3/4, F1-8/Z, FC1-6/Z, FP1/2/Z, FT7/8). "
            "Reduces features from 244 to ~100. "
            "Not set by default — all 61 electrodes are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for saved plots and results JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plots (useful in environments without a display)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────
    print(f"\nLoading data: {args.data}")
    df = load_eeg_data(args.data)
    summarize_dataset(df)

    # ── 2. Build feature matrix ────────────────────────────────────────────
    print(f"\nExtracting features (condition='{args.condition}') ...")

    from src.features import FRONTAL_ELECTRODES

    electrodes = FRONTAL_ELECTRODES if args.frontal_only else None
    electrode_note = "frontal electrodes only" if args.frontal_only else "all 61 electrodes"
    print(f"  Electrode set: {electrode_note}")

    X, y, groups, feature_names = build_feature_matrix(
        df,
        condition=args.condition,
        electrodes=electrodes,
        verbose=True,
    )

    n_subjects = len(set(groups))
    if n_subjects < args.n_folds * 2:
        print(
            f"\nWARNING: Only {n_subjects} unique subjects found. "
            f"Reducing n_folds from {args.n_folds} to {n_subjects // 2}."
        )
        args.n_folds = max(2, n_subjects // 2)

    # ── 3. Cross-validation ────────────────────────────────────────────────
    print(f"\nRunning {args.n_folds}-fold subject-aware cross-validation ...")
    print("  (Each fold holds out a disjoint set of subjects as test set)\n")

    cv_results = cross_validate_subject_aware(
        X, y, groups,
        n_splits=args.n_folds,
        verbose=True,
    )

    # ── 4. Print summary ───────────────────────────────────────────────────
    print_cv_summary(cv_results)

    # ── 5. Save results ────────────────────────────────────────────────────
    results_path = output_dir / "cv_results.json"
    save_results(cv_results, results_path)

    if not args.no_plots:
        print("Saving plots ...")
        plot_confusion_matrix(cv_results, save_path=output_dir / "confusion_matrix.png")
        plot_roc_curve(cv_results,        save_path=output_dir / "roc_curve.png")
        plot_fold_metrics(cv_results,     save_path=output_dir / "fold_metrics.png")

    # ── 6. Print honest summary for documentation ──────────────────────────
    print("\n" + "=" * 60)
    print("  Honest performance summary")
    print("=" * 60)
    print(f"  Condition        : {args.condition}")
    print(f"  CV strategy      : {args.n_folds}-fold GroupKFold (subject-level)")
    print(f"  Feature set      : {electrode_note}")
    print(f"  n_trials         : {len(y)}  (alcoholic={y.sum()}, control={(y==0).sum()})")
    print(f"  n_subjects       : {n_subjects}")
    print()
    print(f"  Accuracy (mean±SD): {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    print(f"  F1     (mean±SD) : {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
    print(f"  ROC-AUC(mean±SD) : {cv_results['mean_roc_auc']:.3f} ± {cv_results['std_roc_auc']:.3f}")
    print()
    print("  Results saved to:", output_dir.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
