# Audit Report, Scientific Cleanup, and Resume Wording

---

## Phase 1: Repository Audit

### Dataset (confirmed valid)
- **16 subjects**: 8 alcoholic (`a`), 8 control (`c`)
- **61 EEG electrodes**, 256 Hz, 1-second trials, ~468 unique trials total
- **3 conditions**: `S1 obj` (single-stimulus), `S2 match`, `S2 nomatch`
- Public UCI dataset (Begleiter et al., 1999) — legitimate neurophysiological data

---

### Problems found in the original code

#### 1. SUBJECT-LEVEL DATA LEAKAGE — SEVERE
**File:** `FINAL CODE/FFT_ML.py`, `Final Code_and_Data/Final Project Code.txt`

`train_test_split` was applied to individual electrode×trial rows with no grouping.
A single subject contributes ~610 rows to the dataset. When these rows are split
randomly, ~427 go to training and ~183 go to testing — but they all came from the
same brain. The model learns subject-specific EEG characteristics (which are very
stable within individuals), not the alcoholism biomarker.

**Effect**: Reported accuracy is inflated. It measures how well the model
memorises subject identity, not how well it generalises to new individuals.

**Fix applied**: GroupKFold cross-validation with `groups = subject_name`.

---

#### 2. ASYMMETRIC CLASS LABELING — SEVERE
**File:** `Code and Data for Saturday, Nov 25/code 1.txt`

```python
features_a = extract_features(data_a_nomatch, ...)  # alcoholic = S2 nomatch
features_c = extract_features(data_c_match,   ...)  # control   = S2 match
```

The classifier is trained on alcoholic + S2_nomatch vs. control + S2_match.
This conflates subject group with stimulus condition. Even a perfect classifier
that ignores subject identity and only detects the paradigm condition would achieve
100% accuracy. Every metric from this version is scientifically invalid.

**Fix applied**: Only S1_obj (a single-stimulus condition) is used for
classification. Both groups are drawn from the same condition.

---

#### 3. MEANINGLESS AVERAGING
**File:** `Code and Data for Saturday, Nov 25/code 1.txt`

```python
for start in range(1, len(selected_data), 60):
    chunk = selected_data.iloc[start:start+60]
    average = chunk[sample_columns].mean()
```

Averages every 60 consecutive rows regardless of subject, trial, or electrode
boundaries. This produces chimeric signals mixing recordings from different
electrodes, trials, and subjects — biologically meaningless.

**Fix applied**: Features are computed per (subject, trial, electrode) group.

---

#### 4. MISMATCH: P300 CLAIM vs. ELECTRODE EXCLUSION
The original report and code comment about P300 (event-related potential ~300ms
post-stimulus, maximal at parietal and central-parietal electrodes). However, the
code explicitly **removes** parietal (P1–P8, PZ), parieto-occipital (PO3–PO8, POZ),
and central-parietal (CP1–CP6, CPZ) electrodes before analysis.

Removing the electrodes where P300 is strongest and then claiming P300-based
analysis is contradictory. No P300 claim can be supported by this pipeline.

**Fix applied**: No P300 claim is made. The analysis is framed as band-power
features for group classification.

---

#### 5. HARDCODED ABSOLUTE PATHS
All code required editing before it could run on any machine.

**Fix applied**: `main.py` uses command-line arguments with sensible defaults.

---

#### 6. NO CROSS-VALIDATION — SINGLE HIGH-VARIANCE ESTIMATE
A single 70/30 split (random_state=42) was the only evaluation. With 149 rows in
`selected_data.csv`, the test set had only ~45 samples. The reported metric is a
single draw from a high-variance distribution.

**Fix applied**: 4-fold GroupKFold with inner 3-fold hyperparameter search.

---

#### 7. INCOMPLETE EVALUATION METRICS
Only accuracy and a classification report were reported. No confusion matrix
visualisation, no ROC-AUC.

**Fix applied**: Confusion matrix, ROC curve, per-fold bar chart, F1, precision,
recall, and AUC are all computed and saved.

---

#### 8. RAW FFT VS. WELCH PSD
The original code computed the raw FFT magnitude of 1-second epochs and took the
values within 8–13 Hz. Raw FFT magnitude on short epochs is noisy (high spectral
variance). Welch's method with overlapping segments gives more stable band-power
estimates.

**Fix applied**: `scipy.signal.welch` is used throughout.

---

### What IS valid in the original project

| Aspect | Status |
|--------|--------|
| Dataset choice (SMNI CMI) | Valid — peer-reviewed, publicly available |
| FFT/frequency domain analysis | Valid approach |
| Alpha band (8–13 Hz) focus | Supported by literature |
| SVM classifier choice | Appropriate |
| StandardScaler normalisation | Correct |
| MNE bandpass/notch filtering in notebook | Correct signal processing |
| Data formatting notebook | Correctly converts raw SMNI format to wide CSV |

---

## Phase 4: Scientific Cleanup

### Corrected project description

> "We developed a machine learning pipeline to classify EEG recordings as belonging
> to individuals with alcohol use disorder or healthy controls, using the SMNI CMI
> dataset (Begleiter et al., 1999; n=16 subjects, 61 electrodes, 256 Hz).
>
> For each one-second EEG trial, we estimated mean spectral power in four frequency
> bands (delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz) per electrode
> using Welch's method, producing a 244-dimensional feature vector per trial.
> Classification was performed using a support vector machine (SVM) with RBF kernel
> and L2 regularisation, with hyperparameters selected via inner cross-validation.
>
> To prevent subject-level data leakage — a common flaw in EEG-ML pipelines where
> trials from the same individual appear in both training and test sets — we used
> 4-fold GroupKFold cross-validation with subject identity as the grouping variable.
> Each fold tested on subjects unseen during training.
>
> Under this honest evaluation, mean accuracy was 53.1% ± 21.1% and mean ROC-AUC
> was 52.7% ± 28.9%, which are near-chance values with high variance. This outcome
> is consistent with the expected difficulty of this problem at n=16 subjects and
> highlights that previously reported higher accuracies in similar pipelines likely
> reflect subject-level leakage rather than true generalisation."

---

### Wording to remove from any writeup

- "P300 event-related potential analysis" — not supported (parietal electrodes were removed)
- Any specific accuracy from the original code — cannot be verified and likely inflated
- "Strong classifier performance" — not supported by corrected analysis
- "Demonstrates that EEG can distinguish alcoholic from control subjects" — overstated for n=16

### Wording to keep / add

- "Subject-aware cross-validation to prevent data leakage"
- "Band-power feature extraction using Welch's method"
- "Near-chance performance under honest subject-held-out evaluation"
- "Highlights the importance of proper CV design in neuroimaging ML"

---

## Phase 5: Final Truth Check — What You Can Honestly Claim

### On GitHub / README

You can say:
- "Implemented a subject-aware EEG classification pipeline to avoid data leakage"
- "Extracted multi-band EEG power features (delta/theta/alpha/beta) across 61 electrodes using Welch's PSD"
- "Evaluated using GroupKFold CV (groups = subjects) — mean accuracy 53% ± 21%, ROC-AUC 53% ± 29%"
- "Identified and corrected subject-level data leakage present in the original prototype"
- "Refactored a monolithic research script into a modular, documented Python pipeline"

You cannot say:
- Any specific high accuracy without noting it came from a leaky evaluation
- "Achieved X% accuracy in classifying alcoholic vs. control EEG" unless X is from the corrected pipeline

---

### On your resume

#### Option A — Concise

> Built a subject-aware EEG classification pipeline (Python, scikit-learn) on a
> public neurophysiological dataset; corrected subject-level data leakage in
> the original design and reported honest cross-validated performance metrics.

#### Option B — Technical

> Developed an EEG-based alcoholism detection pipeline using Welch PSD band-power
> features (delta/theta/alpha/beta, 61 electrodes) and SVM with nested
> GroupKFold cross-validation; identified and resolved subject-level data leakage
> that inflated baseline accuracy, resulting in reproducible mean ROC-AUC of 0.53
> ± 0.29 on n=16 subjects.

#### Option C — Research-oriented

> Investigated EEG alpha-band power as a biomarker for alcohol use disorder
> (SMNI CMI dataset, n=16, 256 Hz, 61 channels); redesigned the ML evaluation
> protocol to enforce subject-held-out cross-validation, exposing inflated accuracy
> estimates in the original pipeline and establishing a reproducible baseline for
> future work with larger cohorts.

---

### Five improved project title options

1. **Subject-Aware EEG Classification: Detecting Alcohol Use Disorder with Honest Cross-Validation**
2. **EEG Band-Power Features and SVM for Alcoholism Detection — A Leakage-Corrected Pipeline**
3. **Reproducible EEG Machine Learning: Correcting Subject-Level Data Leakage in Neurophysiological Classification**
4. **Group-Aware Cross-Validation for EEG-Based Biomarker Detection in Alcohol Use Disorder**
5. **Honest EEG ML: Band-Power Feature Extraction and Subject-Held-Out Evaluation for Alcoholism Classification**

---

### In an interview

If asked about your results:

> "The corrected pipeline achieves near-chance classification performance (accuracy ~53%,
> AUC ~0.53) on subject-held-out folds, which is honest given the dataset has only
> 16 subjects. The original prototype appeared to do much better because it leaked
> subject identity into the test set — the key contribution of this project was
> identifying that flaw and fixing it. The refactored code is modular, reproducible,
> and reports uncertainty alongside point estimates."

This answer demonstrates: understanding of data leakage, scientific integrity,
ML evaluation design, and software engineering — all of which are genuinely valuable
even without a high accuracy number.
