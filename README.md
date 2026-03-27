# EEG-Based Alcoholism Detection: Band-Power Features and Subject-Aware SVM

A machine learning pipeline for classifying EEG signals as belonging to individuals
with alcohol use disorder or healthy controls. Built on a public neurophysiological
dataset using Welch PSD band-power features and subject-aware cross-validation.

---

## Key design principle

Most published EEG-ML pipelines inadvertently inflate their accuracy by splitting
individual electrode×trial rows randomly, placing trials from the same subject in
both training and test sets. This causes **subject-level data leakage** — the model
learns to recognise subjects, not the biomarker of interest.

This pipeline uses **GroupKFold cross-validation** (`groups = subject name`) to
ensure that every test fold contains subjects the model has never seen during
training. This produces honest generalisation estimates.

---

## Results (subject-held-out 4-fold CV on S1 obj condition)

| Metric | Mean | ± SD |
|--------|------|------|
| Accuracy | 0.531 | 0.211 |
| F1 | 0.529 | 0.247 |
| Precision | 0.496 | 0.203 |
| Recall | 0.575 | 0.305 |
| ROC-AUC | 0.527 | 0.289 |

**Interpretation**: Mean performance is near chance (0.50) with very high fold-to-fold
variance (±0.21–0.29). This is the expected honest outcome on a dataset with only
16 subjects (8 per class) when subject identity is kept out of training. The large
standard deviations indicate that some subject subgroups are more separable than others
— a known challenge in small-N neuroimaging studies.

**Why this matters for the resume/portfolio**: The original prototype reported
inflated accuracy (~70–90%+) because it used random row-level splits that leaked
subject information. The corrected analysis shows the actual difficulty of this
problem at this dataset size — which is a more defensible and educational result.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the data

Download and prepare the EEG dataset (see `data/README.md`), or use the included
sample for a smoke test:

```bash
# Smoke test with sample data (4 subjects, 2 trials each)
python main.py --data data/sample/eeg_sample.csv --n-folds 2
```

### 3. Run the full pipeline

```bash
python main.py --data data/EEG_formatted.csv
```

### 4. Options

```
python main.py --help

  --data           Path to EEG CSV (default: data/EEG_formatted.csv)
  --condition      EEG condition: S1 obj | S2 match | S2 nomatch  (default: S1 obj)
  --n-folds        Number of GroupKFold folds (default: 4)
  --frontal-only   Use only frontal electrodes (26 vs 61)
  --output-dir     Where to save plots and results JSON (default: reports/)
  --seed           Random seed (default: 42)
  --no-plots       Skip plot generation
```

---

## Repository structure

```
eeg-alcohol-classification-ml/
├── main.py                  # Entry point — runs the full pipeline
├── requirements.txt
├── src/
│   ├── data_loader.py       # CSV loading and validation
│   ├── preprocessing.py     # Welch PSD band-power extraction
│   ├── features.py          # Feature matrix construction (trial-level)
│   ├── train.py             # Subject-aware CV with inner hyperparameter search
│   ├── evaluate.py          # Metrics, confusion matrix, ROC curve
│   └── utils.py             # Reproducibility, serialisation
├── data/
│   ├── README.md            # How to download the SMNI CMI dataset
│   └── sample/
│       └── eeg_sample.csv   # 4-subject smoke-test fixture
├── notebooks/               # Exploratory notebooks (archived from original project)
├── reports/
│   ├── project_summary.md   # Scientific write-up and limitations
│   ├── cv_results.json      # Full cross-validation results
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── fold_metrics.png
└── archive/                 # Original course project files (BME772)
```

---

## Dataset

**SMNI CMI EEG Alcohol Study** — UCI ML Repository
(Begleiter et al., 1999; 16 subjects, 61 electrodes, 256 Hz, 3 conditions)

See `data/README.md` for download and preparation instructions.

---

## Feature engineering

For each EEG trial (one second of 61-electrode recording):

1. For every electrode, estimate Welch PSD (nperseg = 256)
2. Extract mean power in four bands:
   - Delta (1–4 Hz), Theta (4–8 Hz), **Alpha (8–13 Hz)**, Beta (13–30 Hz)
3. Flatten into a 244-dimensional feature vector (61 × 4)

Alpha-band power suppression is the primary neurophysiological marker motivating
this analysis, consistent with published EEG research on alcohol use disorder.

---

## Model

- **SVM** with RBF kernel (or linear, selected by inner CV)
- Regularisation parameter C searched over {0.01, 0.1, 1, 10, 100}
- StandardScaler fitted per fold on training data only
- Inner GroupKFold (3-fold) nested inside outer GroupKFold (4-fold)

---

## Known limitations

- Only 16 subjects — extremely small for this type of classification study
- No artefact rejection (ICA/EOG) applied beyond average re-referencing
- Single dataset — generalisation to other populations is unknown
- High inter-fold variance makes point estimates unreliable; CIs overlap chance

---

## Original project context

This repository is a refactored and corrected version of a BME772 course project
(archived in `BME772 Project/`). The original prototype used random row-level
train/test splits without subject-level grouping, which inflated reported accuracy.
The original code is preserved as-is for comparison.
