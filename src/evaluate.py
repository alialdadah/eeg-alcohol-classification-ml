"""
evaluate.py
-----------
Metric reporting and visualisation for cross-validation results.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


# ── Console summary ─────────────────────────────────────────────────────────

def print_cv_summary(cv_results: dict) -> None:
    """
    Print a formatted table of cross-validation metrics.

    Parameters
    ----------
    cv_results : dict
        Output of train.cross_validate_subject_aware().
    """
    print("\n" + "=" * 60)
    print("  Subject-Aware Cross-Validation Results")
    print("=" * 60)

    header = f"  {'Fold':>4}  {'Acc':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'AUC':>6}  Test subjects"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in cv_results["fold_results"]:
        subjs = ", ".join(s[-4:] for s in r["test_subjects"])  # last 4 chars of name
        auc_str = f"{r['roc_auc']:.3f}" if not np.isnan(r["roc_auc"]) else "  N/A"
        print(
            f"  {r['fold']:>4}  "
            f"{r['accuracy']:>6.3f}  "
            f"{r['f1']:>6.3f}  "
            f"{r['precision']:>6.3f}  "
            f"{r['recall']:>6.3f}  "
            f"{auc_str:>6}  "
            f"{subjs}"
        )

    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'Mean':>4}  "
        f"{cv_results['mean_accuracy']:>6.3f}  "
        f"{cv_results['mean_f1']:>6.3f}  "
        f"{cv_results['mean_precision']:>6.3f}  "
        f"{cv_results['mean_recall']:>6.3f}  "
        f"{cv_results['mean_roc_auc']:>6.3f}"
    )
    print(
        f"  {'±SD':>4}  "
        f"{cv_results['std_accuracy']:>6.3f}  "
        f"{cv_results['std_f1']:>6.3f}  "
        f"{cv_results['std_precision']:>6.3f}  "
        f"{cv_results['std_recall']:>6.3f}  "
        f"{cv_results['std_roc_auc']:>6.3f}"
    )
    print("=" * 60)
    print()

    # Honest interpretation note
    print("NOTE: Metrics are averaged over subject-held-out folds.")
    print("      Each test fold contains subjects unseen during training.")
    print("      High variance across folds is expected with n=16 subjects.\n")


# ── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cv_results: dict,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the aggregate confusion matrix across all folds.

    Parameters
    ----------
    cv_results : dict
    save_path : str, Path, or None
        If provided, saves the figure.  Otherwise shows it interactively.
    """
    y_true_all = np.concatenate([r["y_test"] for r in cv_results["fold_results"]])
    y_pred_all = np.concatenate([r["y_pred"] for r in cv_results["fold_results"]])

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Control", "Alcoholic"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (aggregated across CV folds)")
    fig.tight_layout()

    _save_or_show(fig, save_path)


# ── ROC curve ───────────────────────────────────────────────────────────────

def plot_roc_curve(
    cv_results: dict,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot per-fold ROC curves and the macro-average.

    Parameters
    ----------
    cv_results : dict
    save_path : str, Path, or None
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    tprs = []
    mean_fpr = np.linspace(0, 1, 200)

    for r in cv_results["fold_results"]:
        if np.isnan(r["roc_auc"]):
            continue
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_proba"])
        fold_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        ax.plot(fpr, tpr, alpha=0.3, lw=1, label=f"Fold {r['fold']} (AUC={fold_auc:.2f})")

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="navy",
            lw=2,
            label=f"Mean ROC (AUC={mean_auc:.2f} ± {cv_results['std_roc_auc']:.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Subject-Held-Out Cross-Validation")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    _save_or_show(fig, save_path)


# ── Bar chart of per-fold metrics ───────────────────────────────────────────

def plot_fold_metrics(
    cv_results: dict,
    save_path: str | Path | None = None,
) -> None:
    """
    Bar chart showing accuracy, F1, and AUC for each fold.
    """
    folds = [r["fold"] for r in cv_results["fold_results"]]
    accs  = [r["accuracy"] for r in cv_results["fold_results"]]
    f1s   = [r["f1"] for r in cv_results["fold_results"]]
    aucs  = [r["roc_auc"] for r in cv_results["fold_results"]]

    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, accs, width, label="Accuracy")
    ax.bar(x,         f1s,  width, label="F1")
    ax.bar(x + width, aucs, width, label="ROC-AUC")
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="Chance")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Fold Performance (Subject-Held-Out CV)")
    ax.legend()
    fig.tight_layout()

    _save_or_show(fig, save_path)


# ── Helper ───────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
