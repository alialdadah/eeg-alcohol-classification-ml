# Documentation

| File | Describes |
|------|-----------|
| [technical_summary.md](technical_summary.md) | Full technical write-up: dataset, methods, results, limitations, extensions |
| [main.md](main.md) | `main.py` — CLI entry point and pipeline orchestration |
| [data_loader.md](data_loader.md) | `src/data_loader.py` — CSV loading and validation |
| [preprocessing.md](preprocessing.md) | `src/preprocessing.py` — Welch PSD band-power computation |
| [features.md](features.md) | `src/features.py` — trial-level feature matrix construction |
| [train.md](train.md) | `src/train.py` — nested GroupKFold cross-validation |
| [evaluate.md](evaluate.md) | `src/evaluate.py` — metrics, confusion matrix, ROC curve |
| [utils.md](utils.md) | `src/utils.py` — seed, JSON I/O, sample CSV generation |
