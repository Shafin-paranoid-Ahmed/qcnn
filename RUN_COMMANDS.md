# Standalone QCNN runbook

Run from this repository root after installing requirements. Set QCNN_DATA_ROOT
as explained in README.md. Existing splits and results are frozen evidence.

```powershell
$PY = '.\.venv\Scripts\python.exe'
# Fast synthetic verification; no dataset required.
& $PY scripts/smoke_test.py
# Explicit one-epoch dataset smoke; writes outputs tagged as smoke.
& $PY scripts/train_qcnn.py --split-seed 0 --smoke --tag standalone_smoke --batch-size 4 --encoding amplitude10
# Full registered clean comparison; run only in a disposable reproduction checkout.
& $PY scripts/run_experiment.py --phase A --split-seeds 0 1 2 --epochs 100 --patience 15 --batch-size 4 --baseline-epochs 100 --encoding amplitude10 --d-conv 2 --reupload-L 1
```

For ablations and attention, use the parameter budgets in
[the historical runbook](docs/historical/ORIGINAL_RUN_COMMANDS.md), replacing its
old interpreter with `$PY` and its `full_qcnn_pilot/scripts/` prefix with
`scripts/`. Historical G:/F: paths are not current setup instructions.
Phase B is unimplemented and must not be represented as a completed experiment.
