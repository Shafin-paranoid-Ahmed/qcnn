# Validation record

## Fresh install (2026-10-13)

A new virtual environment was created with the pinned `requirements.txt` versions
(Torch 2.11.0 CPU build, PennyLane 0.42.3, NumPy 1.26.4) on **Python 3.12**
(Windows 11, CPU only; the documented target is Python 3.10), with PYTHONPATH cleared:

- `scripts/smoke_test.py`: PASS for amplitude10 and amplitude10_mag with attention
  (finite logits and gradients). No files were written to the repository.

## Earlier checks (2026-10-12)

- `scripts/smoke_test.py`: PASS for amplitude10 and amplitude10_mag with attention.
  Synthetic image preprocessing, finite five-class logits, and finite gradients
  for every trainable parameter were checked. No research result was overwritten.
- Result CSV/JSON, figures, checkpoints, split files, and the metrics
  implementation were hash-checked; see [MANIFEST.json](MANIFEST.json).
- All Python files parsed successfully; executable scripts contain no
  machine-specific absolute paths.
- All relative Markdown file links resolved locally.
- Selected paths, sizes, and common credential patterns were reviewed. No matching
  credential patterns were found; this is not an exhaustive security audit.
- A fresh HTTPS clone from GitHub passed the same CPU/synthetic checks with
  PYTHONPATH cleared.

Environment: Python 3.10.20 with NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.2,
Torch 2.11.0+cu128 and PennyLane 0.42.3, in an existing environment.

## Not performed

No full model retraining, dataset regeneration,
full GPU acceptance matrix, or large scaling experiment was performed.

## Notes

CPU unittest discovery also imports the optional GPU module, so the documented
CPU entry point loads the CPU test modules explicitly.
