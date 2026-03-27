# Technical Summary

## EEG-Based Alcoholism Detection Using P300 ERP and Band-Power Features with Subject-Aware SVM

---

## 1. Problem Statement

Classify 1-second EEG recordings as belonging to individuals with **alcohol use disorder (AUD)** or **healthy controls** using spectral features and a support vector machine. The primary scientific motivation is that chronic AUD is associated with measurable changes in EEG rhythms, particularly suppression of alpha-band (8–13 Hz) power.

---

## 2. Dataset

**SMNI CMI EEG Alcohol Study** — UCI Machine Learning Repository (Begleiter et al., 1999)

| Property | Value |
|----------|-------|
| Subjects | 16 total (8 AUD, 8 control) |
| Electrodes | 61 (standard 10-20 montage) |
| Sampling rate | 256 Hz |
| Epoch length | 1 second (256 samples) |
| Trials per subject | ~28–30 per condition |
| Conditions | `S1 obj`, `S2 match`, `S2 nomatch` |
| Total rows in CSV | 28,548 (one row = one electrode × one trial) |
| Total unique trials | ~468 |
| Condition used | `S1 obj` (single-stimulus; no match/nomatch task) |

**Why S1 obj?** The S2 conditions involve a match/nomatch oddball paradigm that elicits the P300 event-related potential — a cognitive response to stimulus novelty. Using S2 conditions for AUD classification risks confounding group differences with paradigm-driven cognitive differences. S1 is a single-stimulus baseline that avoids this confound.

---

## 3. Feature Extraction

### 3.1 Unit of analysis

Each EEG trial is one (subject, trial_number, condition) group. The 61 electrode rows belonging to that trial are processed together and collapsed into a single feature vector.

### 3.2 Method: Welch Power Spectral Density

For each electrode, `scipy.signal.welch` is applied to the 256-sample signal:
- Segment length: 256 samples (= full epoch)
- Overlap: 50% (default)
- Window: Hann
- Frequency resolution: 1 Hz

Mean PSD is computed in each of four bands:

| Band | Range | Notes |
|------|-------|-------|
| Delta | 1–4 Hz | Slow-wave; not the primary focus |
| Theta | 4–8 Hz | Frontal midline theta; some AUD evidence |
| **Alpha** | **8–13 Hz** | **Primary biomarker; suppressed in AUD** |
| Beta | 13–30 Hz | Active processing; higher in some AUD studies |

### 3.3 Feature vector

```
feature vector = [elec_0_delta, elec_0_theta, elec_0_alpha, elec_0_beta,
                  elec_1_delta, ...,
                  elec_60_beta]
```

**Dimensionality:** 61 electrodes × 4 bands = **244 features per trial**

With ~160 trials for the S1 obj condition, this is a moderate-dimensional, small-sample problem — typical for EEG-ML with real neuroimaging datasets.

---

## 4. Machine Learning Pipeline

### 4.1 Model

```
Pipeline:
  StandardScaler → SVC(kernel=rbf|linear, C=?, probability=True)
```

The scaler is fitted on the training partition only (per fold), preventing information leakage through normalisation statistics.

### 4.2 Evaluation: Nested Subject-Aware Cross-Validation

```
Outer CV:  GroupKFold(n_splits=4)
           groups = subject name
           → Each fold: 12 subjects train, 4 subjects test
           → Test subjects are NEVER seen during training

Inner CV:  GroupKFold(n_splits=3)   [on training subjects only]
           Grid search:
             svm__C      ∈ {0.01, 0.1, 1, 10, 100}
             svm__kernel ∈ {'rbf', 'linear'}
           Scored by F1
```

**Why GroupKFold is mandatory here:**
Each subject contributes ~10 trials. Trial rows from the same subject are highly correlated (same brain, same session). A naïve `train_test_split` scatters these correlated rows across train and test, letting the model implicitly learn subject identity. This is called subject-level data leakage and produces inflated, non-generalisable accuracy. GroupKFold prevents it.

### 4.3 Metrics reported

For each outer fold: accuracy, F1, precision, recall, ROC-AUC, confusion matrix entries, best hyperparameters. Final summary: mean ± SD across folds.

---

## 5. Results

All results: 4-fold GroupKFold (subject-held-out), seed=42.

### 5.1 Three-mode comparison

| Feature mode | n_features | Condition | Acc (mean±SD) | F1 | ROC-AUC (mean±SD) |
|---|---|---|---|---|---|
| Band-power only | 244 | S1 obj | 0.531 ± 0.211 | 0.529 | 0.527 ± 0.289 |
| **P300 only** | **20** | **S2 nomatch** | **0.483 ± 0.144** | **0.559** | **0.587 ± 0.090** |
| Combined P300 + band-power | 264 | S2 nomatch | 0.391 ± 0.237 | 0.427 | 0.584 ± 0.353 |

### 5.2 Per-fold detail — P300 only (best mode)

| Fold | Test subjects | Acc | F1 | AUC |
|------|--------------|-----|-----|-----|
| 1 | 0365, 0369, 0372, 0337 | 0.486 | 0.553 | 0.579 |
| 2 | 0370, 0338, 0339, 0341 | 0.459 | 0.622 | 0.617 |
| 3 | 0364, 0371, 0340, 0345 | 0.595 | 0.621 | 0.675 |
| 4 | 0368, 0375, 0342, 0344 | 0.392 | 0.441 | 0.476 |
| **Mean** | | **0.483** | **0.559** | **0.587** |
| **± SD** | | **0.144** | **0.098** | **0.090** |

### 5.3 Interpretation

**P300-only is the most stable and scientifically grounded mode:**

- Lowest AUC variance by a factor of 3 (±0.090 vs ±0.289 for band-power, ±0.353 for combined)
- Highest mean ROC-AUC (0.587) with only 20 features vs 244 or 264
- Consistent with the AUD literature — P300 amplitude reduction is one of the most replicated neurophysiological findings in AUD research

**Why does combined perform worse than P300 alone?**
Adding 244 band-power features to 20 P300 features when training on ~110 samples
per fold causes the SVM to overfit to the high-dimensional band-power block,
drowning out the discriminative P300 signal. This is the **curse of dimensionality**
on a small-n dataset. The combined approach would benefit from PCA or feature
selection before SVM.

**All modes are near chance at this dataset size (n=16).** The honest conclusion is
that 16 subjects is insufficient for a generalisable EEG-based AUD classifier,
regardless of feature choice. P300 is the correct scientific choice; the sample
size is the binding constraint.

---

## 6. Comparison: Original vs. Corrected Pipeline

| Aspect | Original code | Corrected pipeline |
|--------|--------------|-------------------|
| Train/test split | `train_test_split` on rows | `GroupKFold(groups=subject)` |
| Data per sample | 1 electrode row | 1 full trial (61 electrodes) |
| Features | Raw FFT magnitude, 8–13 Hz only | Welch PSD 4 bands + P300 ERP |
| P300 electrodes | Deleted (P, CP, O regions removed) | Restored: CZ, PZ, P3, P4, POZ |
| Subject leakage | Yes | No |
| Condition labeling | Asymmetric (AUD=S2nomatch, ctrl=S2match) | Both classes from same condition |
| Hyperparameter search | None | Inner GroupKFold GridSearchCV |
| Metrics | Accuracy only | Acc, F1, Prec, Rec, ROC-AUC, CM |
| Reproducibility | Hardcoded paths | CLI, seed, modular code |
| Reported accuracy | ~70–90% (leaky, unverified) | 48.3% ± 14.4% (honest, P300 mode) |

---

## 7. Limitations

1. **Sample size (n=16):** Below the minimum for robust neuroimaging ML. All results should be treated as a methodology demonstration, not a scientific finding.

2. **No pre-stimulus baseline:** P300 features are extracted from the raw 250–500 ms window without baseline correction, because the CSV epochs start at stimulus onset. Proper baseline correction would require pre-stimulus samples.

3. **No artefact rejection:** ICA or threshold-based removal of ocular/muscular artefacts was not applied.

4. **Single dataset / single site:** Generalisation to other populations, hardware, or protocols is unknown.

5. **Combined features need dimensionality reduction:** PCA or filter-based feature selection before SVM would likely make combined features competitive with P300-only.

---

## 8. Possible Extensions

| Extension | Expected benefit |
|-----------|-----------------|
| Larger cohort (UCI full SMNI: 122 subjects) | Dramatically reduced variance; publishable results |
| Baseline-corrected P300 (from raw files) | More precise ERP amplitude estimates |
| ICA artefact removal | Cleaner signal throughout |
| PCA before SVM on combined features | Fixes curse of dimensionality for combined mode |
| Connectivity features (coherence, PLV) | Captures inter-regional coupling differences |
| Deep learning (EEGNet, ShallowConvNet) | Learns spatial+temporal features jointly |
| LOSO cross-validation | One subject per test fold — maximum data use |

---

## 9. Repository Structure

```
eeg-alcohol-classification-ml/
├── main.py            ← CLI entry point
├── requirements.txt
├── .gitignore         ← data/EEG_formatted.csv excluded (>100 MB)
├── README.md
├── src/
│   ├── data_loader.py     ← load + validate CSV
│   ├── preprocessing.py   ← Welch PSD band power
│   ├── features.py        ← trial-level feature matrix
│   ├── train.py           ← nested GroupKFold + GridSearchCV
│   ├── evaluate.py        ← metrics + 3 plot types
│   └── utils.py           ← seed, JSON I/O, sample maker
├── data/
│   ├── README.md          ← download instructions for full CSV
│   └── sample/
│       └── eeg_sample.csv ← 4 subjects, smoke-test fixture
├── notebooks/
│   ├── data_formatting.ipynb         ← raw → CSV conversion
│   └── feat_extraction_original.ipynb← original MNE exploration
├── docs/
│   ├── technical_summary.md  ← this file
│   ├── main.md
│   ├── data_loader.md
│   ├── preprocessing.md
│   ├── features.md
│   ├── train.md
│   ├── evaluate.md
│   └── utils.md
└── reports/
    ├── cv_results.json
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── fold_metrics.png
    ├── project_summary.md
    └── audit_and_resume.md
```

---

## 10. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (requires EEG_formatted.csv in data/)
python main.py --data data/EEG_formatted.csv

# Smoke test (no large data needed)
python main.py --data data/sample/eeg_sample.csv --n-folds 2

# See all options
python main.py --help
```
