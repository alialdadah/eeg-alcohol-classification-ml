# Technical Summary

## EEG-Based Alcoholism Detection Using Band-Power Features and Subject-Aware SVM

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

Results are from the S1 obj condition, all 61 electrodes, 4-fold GroupKFold, seed=42.

| Fold | Test subjects | Acc | F1 | AUC |
|------|--------------|-----|-----|-----|
| 1 | 0369, 0375, 0340, 0345 | 0.775 | 0.791 | 0.878 |
| 2 | 0368, 0372, 0339, 0344 | 0.375 | 0.359 | 0.335 |
| 3 | 0365, 0371, 0338, 0342 | 0.275 | 0.216 | 0.165 |
| 4 | 0364, 0370, 0337, 0341 | 0.700 | 0.750 | 0.732 |
| **Mean** | | **0.531** | **0.529** | **0.527** |
| **± SD** | | **0.211** | **0.247** | **0.289** |

### Interpretation

Mean performance is **near chance (0.50 baseline)** with **very high fold-to-fold variance (±0.21–0.29)**. This is the expected honest result for a dataset of only 16 subjects under proper subject-held-out evaluation.

The range from 0.275 to 0.775 across folds reveals that classification success depends heavily on which 4 subjects land in the test set — a direct consequence of the small cohort size. Some subjects may have more pronounced or more detectable AUD-related EEG changes than others; with 8 subjects per class, a single atypical subject dramatically affects fold-level performance.

### What this tells us

- The current feature set and model **cannot reliably generalise** to unseen individuals at this dataset size.
- The result does **not** mean alpha-band power is useless as a biomarker — it means 16 subjects is insufficient to train a generalisable model.
- Published studies showing high accuracy on this dataset using random splits are reporting **subject-recognition performance**, not **AUD classification performance**.

---

## 6. Comparison: Original vs. Corrected Pipeline

| Aspect | Original code | Corrected pipeline |
|--------|--------------|-------------------|
| Train/test split | `train_test_split` on rows | `GroupKFold(groups=subject)` |
| Data per "sample" | 1 electrode row | 1 full trial (61 electrodes) |
| Features | Raw FFT magnitude (8–13 Hz only) | Welch PSD in 4 bands, all 61 electrodes |
| Subject leakage | Yes — same subject in train + test | No — test subjects never seen in training |
| Class labeling | One version used asymmetric conditions | Both classes from same condition (S1 obj) |
| Hyperparameter search | None (default SVM) | Inner GroupKFold GridSearchCV |
| Metrics | Accuracy only | Acc, F1, Precision, Recall, ROC-AUC, CM |
| Reproducibility | Hardcoded paths | CLI args, seed, modular code |
| Reported accuracy | ~70–90% (inflated, unverifed) | 53.1% ± 21.1% (honest, reproducible) |

---

## 7. Limitations

1. **Sample size (n=16):** Below the minimum for robust neuroimaging ML. Modern EEG-ML studies use hundreds to thousands of subjects. Results here should be treated as a methods demonstration, not a scientific finding.

2. **No artefact rejection:** Ocular (blink), muscular, and movement artefacts remain in the signal. The formatted CSV was average-referenced in the preprocessing notebook but ICA or threshold-based rejection was not applied.

3. **Single dataset / single site:** All recordings come from one lab with one protocol and one hardware setup. Generalisation to other populations, protocols, or EEG hardware is unknown.

4. **Feature selection not applied:** 244 features with ~120 training samples per fold is a high-dimensional regime. Dimensionality reduction (PCA, feature selection) was not applied but could reduce overfitting in individual folds.

5. **P300 not analysed:** The P300 event-related potential (a time-domain feature peaking ~300ms post-stimulus at parietal electrodes) is a well-documented AUD biomarker. This pipeline does not compute it. A separate time-domain analysis using S2 nomatch trials would be required.

---

## 8. Possible Extensions

| Extension | Expected benefit |
|-----------|-----------------|
| Larger dataset (UCI full SMNI: 122 subjects) | Dramatically reduced variance; publishable results |
| ICA artefact removal | Cleaner signal; better band-power estimates |
| Time-domain features (P300 amplitude, latency) | Adds a well-validated biomarker |
| Connectivity features (coherence, PLV between electrodes) | Captures inter-regional coupling |
| Deep learning (EEGNet, ShallowConvNet) | Learns spatial+temporal features jointly |
| LOSO cross-validation | Maximum data use; one subject per test fold |

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
