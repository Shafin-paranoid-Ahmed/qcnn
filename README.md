# Patch QCNN

Standalone full-resolution hierarchical patch quantum convolutional classifier,
with B-PCNN and B-QUAD classical controls, magnitude/depth ablations, and attention.

## Research status

Phase A, depth/magnitude ablations, and attention/saliency are complete in the
imported historical results. Phase B corruption evaluation remains a stub.
Best combined QCNN test Macro-F1 was **0.5464 +/- 0.0345**, below B-QUAD
**0.7264 +/- 0.0198**. These results support architectural characterization,
not quantum advantage. See [the reports](results/QCNN_REPORT.md),
[attention report](results/ATTENTION_REPORT.md), and [protocol](docs/protocols/full_qcnn.md).

## Install and verify (PowerShell)

Use Python 3.10, the verified source interpreter version, and a fresh environment:

```powershell
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe scripts/smoke_test.py
```

The smoke test uses a synthetic image and checks preprocessing, forward/backward
passes, and attention. It does not train or alter research results.
On other operating systems use the corresponding environment Python executable.
Dependency pins describe the source environment; see VALIDATION.md for what was
actually checked after extraction.

## Data and reproduction

No raw dataset is bundled. Obtain the original Dacon crop disease data under its
dataset terms. The frozen manifests in `splits/` contain the original five-class
crop-3 task, seeds 0/1/2, drawn exclusively from the trusted `test.csv` pool.
Their filenames describe the original dataset partition, not the model's final
evaluation role; the manifests define train/validation/test within that pool.

Set `QCNN_DATA_ROOT` to a folder containing `test/<id>/<id>.jpg` and, only if
rebuilding splits, `test.csv`. Without this setting the default is `./data`.

```powershell
$env:QCNN_DATA_ROOT = 'D:\datasets\dacon'
& .\.venv\Scripts\python.exe scripts/train_qcnn.py --help
```

Keep the shipped split manifests unchanged for reproduction. See
[RUN_COMMANDS.md](RUN_COMMANDS.md) for deliberate training commands. Research
runs can take substantial time and overwrite outputs when names are reused;
use a disposable checkout for a full rerun.

## Contents and provenance

- `scripts/`: models, preprocessing, baselines, training, and evaluation.
- `splits/`: frozen original manifests.
- `results/`: tracked historical metrics, tables, and figures.
- `checkpoints/`: small checkpoints that were tracked in the source snapshot.
- `docs/protocols/`: original specifications; historical paths are provenance.

[PROVENANCE.md](PROVENANCE.md) records the source commit, contributors, and
portability changes; [NOTICE.md](NOTICE.md) records licensing status.
No sibling QCNN/QEMCMC or thesis checkout is required.
