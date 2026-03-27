"""
main.py
-------
Entry point for the EEG Alcohol Classification pipeline.

Usage
-----
    # Combined P300 + band-power (recommended)
    python main.py --data data/EEG_formatted.csv

    # Band-power only on S1 obj
    python main.py --data data/EEG_formatted.csv --features band_power --condition "S1 obj"

    # P300 only
    python main.py --data data/EEG_formatted.csv --features p300

    # Smoke test with sample data
    python main.py --data data/sample/eeg_sample.csv --n-folds 2

The script will:
  1. Load and validate the EEG data.
  2. Build the feature matrix (mode selected by --features).
  3. Run subject-aware GroupKFold cross-validation with inner hyperparameter search.
  4. Print a metrics table and save plots + results JSON to reports/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_loader import load_eeg_data, summarize_dataset
from src.features import (
    build_feature_matrix,
    build_p300_features,
    build_combined_features,
    FRONTAL_ELECTRODES,
)
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
        "--features",
        type=str,
        default="combined",
        choices=["band_power", "p300", "combined"],
        help=(
            "Feature extraction mode:\n"
            "  combined   — P300 ERP features (parietal electrodes) + band-power\n"
            "               features (all electrodes), both from S2 nomatch trials.\n"
            "               Scientifically strongest: uses the validated P300 biomarker\n"
            "               alongside spectral power differences. (default)\n"
            "  band_power — Welch PSD in delta/theta/alpha/beta across all electrodes.\n"
            "               Condition controlled by --condition.\n"
            "  p300       — P300 ERP features only at parietal/central electrodes.\n"
            "               Forces S2 nomatch condition.\n"
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="S1 obj",
        choices=["S1 obj", "S2 match", "S2 nomatch"],
        help=(
            "Condition for band_power mode only. Ignored for p300 and combined "
            "(those always use S2 nomatch, which is required for P300 elicitation)."
        ),
    )
    parser.add_argument(
        "--frontal-only",
        action="store_true",
        help=(
            "For band_power and combined modes: restrict band-power electrodes to "
            "frontal only (26 electrodes instead of 61). Has no effect on the P300 block."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help=(
            "Number of outer GroupKFold folds. Each fold holds out a disjoint set "
            "of subjects. Must be ≤ min(n_alcoholic_subjects, n_control_subjects)."
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
    print(f"\nFeature mode: {args.features.upper()}")
    bp_electrodes = FRONTAL_ELECTRODES if args.frontal_only else None
    bp_note = "frontal electrodes" if args.frontal_only else "all 61 electrodes"

    if args.features == "band_power":
        print(f"  Condition     : {args.condition}")
        print(f"  Electrodes    : {bp_note}")
        X, y, groups, feature_names = build_feature_matrix(
            df,
            condition=args.condition,
            electrodes=bp_electrodes,
            verbose=True,
        )
        mode_description = f"band-power ({bp_note}, {args.condition})"

    elif args.features == "p300":
        print("  Condition     : S2 nomatch  (forced — P300 requires oddball paradigm)")
        print("  P300 electrodes: CZ, PZ, P3, P4, POZ")
        X, y, groups, feature_names = build_p300_features(df, verbose=True)
        mode_description = "P300 ERP features (CZ, PZ, P3, P4, POZ)"

    else:  # combined
        print("  Condition     : S2 nomatch  (forced — P300 requires oddball paradigm)")
        print("  P300 electrodes: CZ, PZ, P3, P4, POZ  (parietal / central-midline)")
        print(f"  Band-power    : {bp_note}")
        X, y, groups, feature_names = build_combined_features(
            df,
            bp_electrodes=bp_electrodes,
            verbose=True,
        )
        mode_description = f"combined P300 + band-power ({bp_note})"

    # ── 3. Auto-adjust n_folds ─────────────────────────────────────────────
    n_subjects = len(set(groups))
    if n_subjects < args.n_folds * 2:
        new_folds = max(2, n_subjects // 2)
        print(
            f"\nWARNING: Only {n_subjects} subjects found. "
            f"Reducing n_folds {args.n_folds} → {new_folds}."
        )
        args.n_folds = new_folds

    # ── 4. Cross-validation ────────────────────────────────────────────────
    print(f"\nRunning {args.n_folds}-fold subject-aware cross-validation ...")
    print("  Each fold holds out a disjoint set of subjects as the test set.\n")

    cv_results = cross_validate_subject_aware(
        X, y, groups,
        n_splits=args.n_folds,
        verbose=True,
    )

    # ── 5. Print + save results ────────────────────────────────────────────
    print_cv_summary(cv_results)
    save_results(cv_results, output_dir / "cv_results.json")

    if not args.no_plots:
        print("Saving plots ...")
        plot_confusion_matrix(cv_results, save_path=output_dir / "confusion_matrix.png")
        plot_roc_curve(cv_results,        save_path=output_dir / "roc_curve.png")
        plot_fold_metrics(cv_results,     save_path=output_dir / "fold_metrics.png")

    # ── 6. Honest summary ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Honest performance summary")
    print("=" * 62)
    print(f"  Feature mode     : {mode_description}")
    print(f"  CV strategy      : {args.n_folds}-fold GroupKFold (subject-level)")
    print(f"  n_trials         : {len(y)}  "
          f"(alcoholic={y.sum()}, control={(y==0).sum()})")
    print(f"  n_subjects       : {n_subjects}")
    print(f"  n_features       : {X.shape[1]}")
    print()
    print(f"  Accuracy  (mean±SD): {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    print(f"  F1        (mean±SD): {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
    print(f"  ROC-AUC   (mean±SD): {cv_results['mean_roc_auc']:.3f} ± {cv_results['std_roc_auc']:.3f}")
    print()
    print("  Results saved to:", output_dir.resolve())
    print("=" * 62)


if __name__ == "__main__":
    main(sys.argv[1:])
