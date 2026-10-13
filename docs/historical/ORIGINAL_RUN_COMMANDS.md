> Historical research record. Original machine paths and commands are retained as provenance. Use the standalone [README](../../README.md) and runbook for current setup.

# Run Commands

Status as of 2026-09-07: Phase A, A2 depth/magnitude/combined ablations, and
A3 attention/saliency are complete. The blocks below preserve the original
reproduction commands and machine paths; they are not pending runs. Phase B
remains an unimplemented stub, and the Level-2 attention trigger was not met.
Read [README](ORIGINAL_README.md) and [RESULTS_LOG](../../RESULTS_LOG.md) before any rerun.
The old unpinned package-install line is historical; use the repository's
pinned environment and do not run it merely to inspect completed results.

```powershell
$PY = "WORKSPACE\.venv-e1\Scripts\python.exe"
cd "WORKSPACE"
```

## Phase A

```powershell
# Splits (default task crop3_disease). Other tasks: healthy | species | stage
& $PY .\full_qcnn_pilot\scripts\build_splits.py --task crop3_disease --split-seeds 0 1 2 --check-images

# Resolution sanity — first real number
& $PY .\full_qcnn_pilot\scripts\phase0_resolution.py --cap-per-class 150 --resolutions 28 56 112 224

# Quantum stack
& $PY -m pip install pennylane pennylane-lightning

# Budget smoke BEFORE full training (spec 7): one clean epoch, seed 0
& $PY .\full_qcnn_pilot\scripts\train_qcnn.py --split-seed 0 --smoke --batch-size 4 --encoding amplitude10

# Full three-seed clean Phase-A run with required B-PCNN and B-QUAD controls
& $PY -u .\full_qcnn_pilot\scripts\run_experiment.py --phase A --split-seeds 0 1 2 --epochs 100 --patience 15 --batch-size 4 --baseline-epochs 100 --encoding amplitude10 --d-conv 2 --reupload-L 1
```

## Phase B (only after Phase A clears go/no-go)

```powershell
# & $PY .\full_qcnn_pilot\scripts\run_experiment.py --task crop3_disease --split-seeds 0 1 2 --phase B --corruptions gaussian_noise defocus_blur motion_blur contrast jpeg --severities 2 3 5
```

## Phase A2 ablations — research PC

```powershell
$PY = "WORKSPACE\.venv-e1\Scripts\python.exe"
Set-Location -LiteralPath "WORKSPACE"

# Completed budget smokes (seed 0 only)
& $PY -u .\full_qcnn_pilot\scripts\train_qcnn.py --split-seed 0 --encoding amplitude10 --reupload-L 2 --tag smoke_L2_amplitude10 --batch-size 4 --lr 0.01 --smoke
& $PY -u .\full_qcnn_pilot\scripts\train_qcnn.py --split-seed 0 --encoding amplitude10 --reupload-L 3 --tag smoke_L3_amplitude10 --batch-size 4 --lr 0.01 --smoke

# Manually launched full A2.1 runs; do not run concurrently.
0,1,2 | ForEach-Object { & $PY -u .\full_qcnn_pilot\scripts\train_qcnn.py --split-seed $_ --encoding amplitude10 --reupload-L 2 --tag L2_amplitude10 --epochs 100 --patience 15 --batch-size 4 --lr 0.01 }
0,1,2 | ForEach-Object { & $PY -u .\full_qcnn_pilot\scripts\train_qcnn.py --split-seed $_ --encoding amplitude10 --reupload-L 3 --tag L3_amplitude10 --epochs 100 --patience 15 --batch-size 4 --lr 0.01 }
```

## Phase A3 attention — research PC

Run only after `Get-Process python*` confirms no training process is alive.

```powershell
$PY = "WORKSPACE\.venv-e1\Scripts\python.exe"

# Required budget smoke; stop if the reported complete epoch exceeds 1200 s.
& $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm qcnn_attn --split-seed 0 --smoke

# Full protocol, only after the smoke passes.
'qcnn_attn','bpcnn_attn','battn' | ForEach-Object { $arm = $_; 0,1,2 | ForEach-Object { & $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm $arm --split-seed $_ --epochs 100 --patience 15 --batch-size 4 --lr 0.01 } }

# Seed-0 saliency for every attention arm and final aggregation.
'qcnn_attn','bpcnn_attn','battn' | ForEach-Object { & $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm $_ --split-seed 0 --saliency }
& $PY .\full_qcnn_pilot\scripts\finalize_attention.py
```

## Phase A2.3 conditional combined runs — HOME PC — manual launch only

The pre-registered trigger fired on 2026-07-14. Run these sequentially, never
concurrently. This block was not executed during implementation.

    $PY = "WORKSPACE\.venv-e1\Scripts\python.exe"
    # A2.3 official runs — sequential, never concurrent
    0,1,2 | ForEach-Object { & $PY -u .\full_qcnn_pilot\scripts\train_qcnn.py --split-seed $_ --encoding amplitude10_mag --reupload-L 2 --tag L2_amplitude10_mag --epochs 100 --patience 15 --batch-size 4 --lr 0.01 }

## Phase A3 permitted smoke - HOME PC - completed during implementation (2026-07-15)

Historical command record only; do not rerun as part of the official matrix.
This was the sole permitted one-epoch smoke and completed below the budget gate.

```powershell
& "WORKSPACE\.venv-e1\Scripts\python.exe" -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm qcnn_attn --split-seed 0 --smoke
```

## Phase A3 attention - HOME PC - manual launch only (2026-07-15)

Run sequentially. Do not run any two arms or seeds concurrently. This is the
official Phase-A3 protocol; no command in this block was executed during implementation.

```powershell
$PY = "WORKSPACE\.venv-e1\Scripts\python.exe"
Set-Location -LiteralPath "WORKSPACE"

# QCNN-attn: fixed amplitude10_mag L=1 base, seeds 0/1/2.
0,1,2 | ForEach-Object { & $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm qcnn_attn --split-seed $_ --epochs 100 --patience 15 --batch-size 4 --lr 0.01 }

# B-PCNN-attn: identical five-parameter gate, seeds 0/1/2.
0,1,2 | ForEach-Object { & $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm bpcnn_attn --split-seed $_ --epochs 100 --patience 15 --batch-size 4 --lr 0.01 }

# B-ATTN attention-only null, seeds 0/1/2.
0,1,2 | ForEach-Object { & $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm battn --split-seed $_ --epochs 100 --patience 15 --batch-size 4 --lr 0.01 }

# Seed-0 saliency exports for every arm, after all nine training runs finish.
& $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm qcnn_attn --split-seed 0 --saliency
& $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm bpcnn_attn --split-seed 0 --saliency
& $PY -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm battn --split-seed 0 --saliency

# Aggregate new rows with reused anchors and render the final report.
& $PY -u .\full_qcnn_pilot\scripts\finalize_attention.py
```

**No official run was launched during implementation.**
