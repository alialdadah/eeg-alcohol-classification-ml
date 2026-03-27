# Project Summary: EEG-Based Alcoholism Detection Using Band-Power Features and SVM

## Overview

This project applies machine learning to classify EEG (electroencephalography) recordings
as belonging to individuals with alcohol use disorder or healthy controls, using a
publicly available, peer-reviewed neurophysiological dataset.

## Dataset

**SMNI CMI EEG Alcohol Dataset** (UCI ML Repository, Begleiter et al., 1999)

- 16 subjects: 8 with chronic alcohol use disorder, 8 matched controls
- ~468 one-second EEG trials across three stimulus conditions
- 61 electrodes, 256 Hz sampling rate
- Condition used: `S1 obj` (single-stimulus paradigm)

Note on dataset size: 16 subjects is a small cohort by modern standards. Results
should be interpreted with appropriate uncertainty; cross-validation standard
deviations are reported alongside means.

## Methodology

### Feature Extraction

For each EEG trial, we compute **Welch power spectral density (PSD)** estimates
for each of the 61 electrodes and extract mean power in four canonical frequency bands:

| Band | Range | Neuroscientific rationale |
|------|-------|--------------------------|
| Delta | 1–4 Hz | Slow cortical activity |
| Theta | 4–8 Hz | Working memory, frontal midline |
| **Alpha** | **8–13 Hz** | **Primary marker of interest: alpha suppression is documented in alcohol use disorder** |
| Beta | 13–30 Hz | Active processing, arousal |

This yields a 244-dimensional feature vector per trial (61 electrodes × 4 bands).

**Why Welch's method over raw FFT?**
The original code used a raw FFT magnitude on 1-second epochs. Welch's method
averages overlapping periodogram segments, which reduces spectral variance and
produces more stable band-power estimates — especially important for the short
epochs in this dataset.

### Classification

- **Algorithm**: Support Vector Machine (SVM) with RBF kernel and L2 regularization
- **Preprocessing**: StandardScaler (zero mean, unit variance), fitted on training data only
- **Hyperparameter search**: Inner GridSearchCV over {C, kernel} nested within outer CV

### Evaluation: Subject-Aware Cross-Validation

**Critical design decision**: The original project split individual electrode×trial
rows randomly (standard `train_test_split`). Because multiple rows from the same
subject are statistically correlated, this causes *subject-level data leakage* — the
model learns subject identity rather than alcoholism biomarkers, producing inflated
accuracy estimates that do not generalise to new individuals.

The corrected pipeline uses **GroupKFold cross-validation** with `groups = subject_name`.
This guarantees that every test fold contains subjects not seen during training,
producing an honest estimate of generalisation to new individuals.

**Evaluation metrics reported**:
- Accuracy
- F1 score
- Precision, recall
- ROC-AUC
- All reported as mean ± SD across folds

## Results

Results are written to `reports/cv_results.json` after running `main.py`.
See also `reports/confusion_matrix.png`, `reports/roc_curve.png`, `reports/fold_metrics.png`.

> **Note**: Performance metrics are not hard-coded in this summary because they depend
> on the actual data and pipeline execution. Run `python main.py` to obtain the
> true, reproducible results. Any metrics cited elsewhere (resume, README, portfolio)
> should be copied from the actual `cv_results.json` output.

## Limitations

1. **Small cohort (n=16)**: With only 8 subjects per class, variance across CV folds
   is high. Results should not be over-interpreted.

2. **No spatial filtering applied**: The pipeline does not include ICA or other
   artefact removal. The formatted CSV was average-referenced in the preprocessing
   notebook, but ocular/muscular artefacts remain.

3. **Single dataset**: All subjects are from one study site. Generalisation to
   other populations, recording hardware, or protocols is unknown.

4. **P300 / event-related potential analysis**: A P300-based analysis would require
   time-domain features (amplitude ~300ms post-stimulus) at parietal electrodes.
   This pipeline uses frequency-domain features and does not claim P300 findings.

## What Can Be Honestly Claimed

- "Applied subject-aware cross-validation to prevent data leakage in an EEG classification task"
- "Extracted multi-band EEG power features (delta/theta/alpha/beta) using Welch's method across 61 electrodes"
- "Achieved [X]% accuracy / [Y] F1 / [Z] ROC-AUC (mean ± SD across 4-fold subject-held-out CV)" — fill from actual run
- "Identified alpha-band power as a candidate feature for distinguishing alcohol use disorder in this dataset"
- "Refactored a prototype ML pipeline into a modular, reproducible Python package"
