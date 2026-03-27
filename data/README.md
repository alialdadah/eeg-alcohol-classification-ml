# Data

## Dataset: SMNI CMI EEG Alcohol Study

**Source:** UCI Machine Learning Repository
**Original study:** Snyder, Steinhauer, Berger (1995); Begleiter et al. (1999)
**UCI page:** https://archive.ics.uci.edu/dataset/121/eeg+database

### What the dataset contains

- **16 subjects**: 8 with alcohol use disorder (identifier `a`), 8 control (`c`)
- **3 conditions**: `S1 obj` (single stimulus), `S2 match`, `S2 nomatch`
- **61 EEG electrodes** (standard 10-20 montage)
- **256 Hz sampling rate**, 1 second (256 samples) per trial
- Approximately **28,500 electrode × trial rows** in the formatted version

### How to obtain and prepare the data

1. Download the SMNI CMI training set from UCI:

   ```
   https://archive.ics.uci.edu/static/public/121/eeg+database.zip
   ```

2. Extract and locate the `SMNI_CMI_TRAIN/` directory.

3. Run the preprocessing notebook to convert the raw files to the
   wide-format CSV used by this pipeline:

   ```
   notebooks/data_formatting.ipynb
   ```

   This will produce `data/EEG_formatted.csv` (~132 MB).

4. Alternatively, if you already have `EEG_formatted.csv`, place it here:

   ```
   data/EEG_formatted.csv
   ```

### Sample data (for testing)

`data/sample/eeg_sample.csv` — 488 rows from 4 subjects (2 alcoholic, 2 control),
2 trials each, all 61 electrodes, condition `S1 obj`. Sufficient to run the
pipeline and verify it executes correctly. **Not suitable for drawing
scientific conclusions.**

### Column schema of EEG_formatted.csv

| Column | Description |
|--------|-------------|
| `name` | Subject identifier string (e.g. `co2a0000364`) |
| `trial number` | Trial index within the session |
| `matching condition` | `S1 obj`, `S2 match`, or `S2 nomatch` |
| `sensor position` | Electrode label (e.g. `AF3`, `CZ`) |
| `subject identifier` | `a` = alcoholic, `c` = control |
| `sample_0`–`sample_255` | EEG voltage values (μV) at each time point |
