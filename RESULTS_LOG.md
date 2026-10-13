> Historical research record. Original machine paths and commands are retained as provenance. Use the standalone [README](README.md) and runbook for current setup.

# Results Log

Dated run notes: commands, artifacts, metrics, interpretation, per-epoch
wall-clock, and go/no-go status.

## 2026-07-11 — Scaffold + splits + Phase-0 resolution

Task `crop3_disease` (5 classes): `3_00_0`(1177), `3_b3_1`(166), `3_b6_1`(142),
`3_b7_1`(156), `3_b8_1`(153); total 1794.

`build_splits.py --check-images`: all 1794 images resolve at
`Data/test/<id>/<id>.jpg`. Splits (per seed): train 1077 / val 358 / test 359.

`phase0_resolution.py --cap-per-class 100 --resolutions 28 56 112`
(logistic regression on flattened raw pixels, train ~471, val 358):

| resolution | val Macro-F1 |
| --- | ---: |
| 28  | 0.5724 |
| 56  | 0.5611 |
| 112 | 0.5698 |

**Interpretation (honest):**
- A LINEAR model on raw pixels plateaus at ~0.57 Macro-F1 and is essentially
  FLAT across resolution. A linear classifier has no locality/texture inductive
  bias, so it cannot exploit extra resolution — this does NOT prove resolution is
  useless; it motivates a model WITH spatial structure (CNN / patch-QCNN) to use
  fine detail. (Run a small CNN at each resolution to see the resolution-sensitive
  curve; the linear curve is the floor.)
- **Reference floor for the QCNN: ~0.57.** Recall an amplitude-encoded QCNN is
  ~quadratic-in-pixels (spec 5.4); a quadratic model is only slightly richer than
  linear. So there is a real risk the QCNN lands near this 0.57 floor unless the
  hierarchical pooling + data re-uploading add genuine nonlinearity. **B-QUAD is
  exactly the control that will detect this.** Watch for it.
- TODO: add res=224 (heavier), and add a small-CNN Phase-0
  row so the resolution-sensitivity of a spatial model is on record.

Status: scaffold verified end-to-end (splits + Phase 0 run clean). Quantum scripts
are stubs — implement per spec, Phase A first.

## 2026-07-11 — Primary QCNN path and budget pre-check

Commands run:

```powershell
& $PY .\full_qcnn_pilot\scripts\build_splits.py --task crop3_disease --check-images
& $PY .\full_qcnn_pilot\scripts\phase0_resolution.py --split .\full_qcnn_pilot\splits\split_seed0.json
```

Verified trusted-label output: all 1,794 selected images resolve; each seed has
1,077 train / 358 val / 359 test samples. Phase-0 validation Macro-F1: 28=0.6003,
56=0.6066, 112=0.5727, 224=0.5605. This linear floor does not show a resolution
gain on `crop3_disease`; it is retained as an honest result.

Implemented and compiled the primary 10-qubit raw-pixel QCNN. One-image output
shape is `[1, 5]`; trainable parameter count is 133. A cold forward took 18.9 s
and a warm forward took 2.957 s on `lightning.qubit`. At 1,077 training images,
the warm forward alone projects to about 53 minutes per epoch before gradients.

**Pre-smoke interpretation:** the primary timing estimate is concerning, so the
Section 5.3 8-qubit-per-channel fallback was implemented and benchmarked next.
This does not replace the handoff-required one-epoch gradient smoke, which still
requires `train_qcnn.py`. No test-set model selection or Phase B work occurred.

## 2026-07-11 — Fallback and CPU-thread budget checks

Implemented and compiled `channel8`, the required 8-qubit-per-channel fallback.
It produced valid `[1, 5]` output with 125 trainable parameters, but its warm
one-image forward time was **5.346 s**, slower than the 10-qubit primary because
it evaluates three channel registers per patch.

A controlled primary-path check with `OMP_NUM_THREADS=8` and
`MKL_NUM_THREADS=8` reduced warm forward time only from 2.957 s to **2.657 s**.
For the 1,077-image `crop3_disease` train split, that is about **47.7 minutes of
forward-only work per epoch**, before adjoint gradients, pooling gradients,
validation, or checkpointing. Both encodings therefore fail the <=20-minute
Section-7 training budget.

**Current state:** these are pre-smoke indicators only, not a Phase-A go/no-go
verdict and not a classification result. The next handoff-required work is to
implement `train_qcnn.py`, `baselines.py`, `metrics.py`, and `run_experiment.py`,
then run the actual one-epoch gradient smoke before any full training decision.

## 2026-07-11 — Required seed-0 one-epoch smoke

Command:

```powershell
& $PY .\full_qcnn_pilot\scripts\train_qcnn.py --split-seed 0 --smoke --batch-size 4 --encoding amplitude10
```

Completed clean training epoch on the trusted seed-0 train split and validation
evaluation only. `results/qcnn_seed0_history.csv` records wall time **173.065 s**,
train loss 1.5482, mean gradient norm 0.8601, validation Macro-F1 **0.2403**, and
validation accuracy 0.6760. The 20-minute Section-7 budget gate is therefore
cleared for this configuration. This is a smoke number, not a held-out test result
and not a Phase-A go/no-go verdict. Next: implement clean fair baselines and the
orchestrator before committing to multi-epoch, three-seed training.

## 2026-07-11 — Corrected default.qubit smoke and calibration audit

Confirmed PennyLane 0.42.3, Torch 2.11.0+cu128, and `crop3_disease` split files
for seeds 0, 1, and 2. Re-ran the required seed-0 smoke after confirming the
measured `default.qubit` + `backprop` broadcast path. The complete train plus
validation epoch took **156.3 seconds** (2.61 minutes), with train loss 1.5482,
mean gradient norm 0.8601, validation Macro-F1 0.2403, and validation accuracy
0.6760. This clears the 20-minute gate without downscaling.

Calibration audit: QCNN and B-PCNN use class-weighted cross-entropy for training
but report ECE/Brier/NLL from their genuine, unreweighted softmax outputs. B-QUAD
uses its balanced logistic model for classification and a second logistic model
fit with `class_weight=None` solely for calibration probabilities.

## 2026-07-11 — Simulation-budget UNBLOCKED via default.qubit broadcast

The Section-7 budget failure was caused solely by `lightning.qubit` executing the
196 per-patch circuits in a Python loop. Switched all three qnodes in
`scripts/qcnn_model.py` from `lightning.qubit`/`adjoint` to
`default.qubit`/`backprop`, which broadcasts the 196 patches as one vectorized
tensor op.

Measured (CPU, this environment):
- Patch layer, 196 patches, 10 qubits, d_conv=2, L=1:
  forward 60 ms (broadcast) vs 2,770 ms (lightning per-sample loop) = **46x**.
- Full PatchQCNN (patch + pool1 + pool2 + backward), batch 8:
  fwd 382 ms, fwd+bwd 1,180 ms → **147 ms/image**.
- Projected TRAIN epoch @ 1,077 imgs, batch 8: **~2.7 min** (fwd+bwd), vs the
  earlier ~48 min forward-only. Well under the 20-min budget.

Output shape [8,5], 133 trainable params — unchanged. No task or architecture
downscale needed. Conclusion: the base Phase-A architecture IS trainable in this
environment. Next: implement `train_qcnn.py` (data caching + loop + early stop)
and run the 1-epoch smoke, then full Phase-A. No training/eval/baseline run yet.

## Phase-A clean go/no-go

QCNN minus B-PCNN test Macro-F1: 0.010592. Verdict: **competitive-clean**.

## 2026-07-11 — Superseded provisional three-seed clean Phase A

Ran the fixed `default.qubit` + `backprop` QCNN, exact 133-parameter B-PCNN,
and B-QUAD on trusted `Data/test.csv`-derived splits for seeds 0, 1, and 2.
Best QCNN checkpoints were selected only by validation Macro-F1; each seed's
held-out test partition was evaluated after selection. Per-seed histories,
checkpoints, `all_runs.csv`, `summary.csv`, `QCNN_REPORT.md`, `scalability.csv`,
and `function_class.md` were generated.

Clean test Macro-F1 mean +/- sample std:

- QCNN: **0.4481 +/- 0.0637**
- B-PCNN: **0.4376 +/- 0.0721**
- B-QUAD: **0.7264 +/- 0.0198**

QCNN exceeds B-PCNN by 0.0106 (1.06 percentage points), so it passes the clean
competitiveness gate of being within approximately three points. This is a clean
Phase-A go for architectural competitiveness only, not evidence of quantum
advantage.

QCNN trails B-QUAD by 0.2783 (27.83 percentage points). It therefore does not
beat the explicit quadratic null and provides no evidence, in this L=1 run, of a
richer-than-quadratic learned function. Its Macro-F1 is also below the earlier
approximately 0.57 linear-resolution floor. The honest result is that the QCNN
is competitive with its parameter-matched hierarchical classical twin but is
substantially weaker than the quadratic baseline. No Phase-B corruption work or
post-result circuit tuning was performed.

This entry is superseded by the corrected B-PCNN rerun below. Review found that
the provisional B-PCNN reduced each patch to channel means before its learned
filter, so it was not an equivalent raw-spatial patch twin.

## 2026-07-11 — Corrected fair B-PCNN rerun and final Phase-A verdict

Replaced B-PCNN with an exact 133-parameter spatial patch CNN that consumes all
`16x16x3` raw normalized pixels per patch. Its shared learned filter is a
depthwise `2x2` convolution followed by pointwise channel mixing and a shared
4-to-4 patch mixer. It retains four features through Level 1 and eight global
features before the linear head. Training now matches QCNN batch size, Adam,
cosine schedule, class-weighted loss, validation patience, genuine-softmax
calibration, and per-seed checkpoint/history output. QCNN and B-QUAD results were
preserved; corrected B-PCNN was rerun for seeds 0, 1, and 2.

Final clean test Macro-F1 mean +/- sample std:

- QCNN: **0.4481 +/- 0.0637**
- B-PCNN: **0.2922 +/- 0.0078**
- B-QUAD: **0.7264 +/- 0.0198**

QCNN exceeds the corrected parameter-matched B-PCNN by 0.1560 (15.60 points),
so the clean competitiveness gate passes. QCNN still trails B-QUAD by 0.2783
(27.83 points). Therefore the quantum hierarchy outperforms its highly
parameter-constrained spatial classical twin, but it does **not** beat the
quadratic null and does not demonstrate richer-than-quadratic benefit at L=1.
No Phase-B, attention, corruption, extra baseline, or post-result tuning was run.

## Phase-A clean go/no-go

QCNN minus B-PCNN test Macro-F1: 0.155966. Verdict: **competitive-clean**.

## 2026-07-11 — Option A `amplitude10_mag` smoke (research PC)

Ran the pre-registered seed-0 one-epoch smoke with `default.qubit`,
`diff_method="backprop"`, batch size 4, d_conv=2, and L=1 on this lab research
PC. The complete training-plus-validation epoch took **83.4 seconds (1.39
minutes)**, with training loss 1.5518 and validation Macro-F1 0.2803. The model
has exactly 133 trainable parameters. This clears the fixed 20-minute-per-epoch
budget without any task or architecture downscaling, so the unchanged
three-seed protocol may proceed.

## 2026-07-12 — Phase A2 state reconciliation and A2.1 budget smokes

Step 0 added configuration-derived output tags to `train_qcnn.py`; new outputs
use `qcnn_{tag}_seed{N}` and the frozen Phase-A result files retained their
original SHA-256 hashes and mtimes. On this lab research PC
(`WORKSPACE`; Python 3.10.11, Torch 2.11.0+cu128, PennyLane 0.42.3,
RTX 4090 present), both `default.qubit` + `backprop` A2.1 budget smokes passed:

- L=2, tag `smoke_L2_amplitude10`: **149.9 s/epoch** (2.50 min), 173 params,
  validation Macro-F1 0.3202.
- L=3, tag `smoke_L3_amplitude10`: **140.2 s/epoch** (2.34 min), 213 params,
  validation Macro-F1 0.1585.

Exact commands are recorded in `RUN_COMMANDS.md`. Both are below the fixed
20-minute gate. No full L=2/L=3 seed was launched during implementation; those commands
are provided for manual execution. A2.1 is a scaling/function-class study,
not a parameter-matched comparison: only L=1 retains the 133-parameter match.

Magnitude-wire reconciliation: the authoritative 2026-07-11 §5.5 design uses
the four fixed input rotations on readout wires 6–9. The completed L=1
`amplitude10_mag` seeds conform and are the official A2.2 runs; no rewiring or
rerun is permitted. Their test Macro-F1 values are 0.497537, 0.542873, and
0.592190, giving **0.544200 +/- 0.047340** (sample std). This improves over the
frozen 0.4481 QCNN mean by 0.0961 and closes 34.53% of the original 0.2783
QCNN-to-B-QUAD gap, while still trailing B-QUAD 0.7264. The fixed channel means
and norm remain arithmetic on normalized raw pixels, with no learned encoder.
A2.2 therefore clears its half of the A2.3 trigger; A2.3 remains pending until
the full A2.1 L=2/L=3 means exist. No baseline, Phase B, attention, or A2.3 run
was launched.

## 2026-07-12 — Phase A3 implementation checkpoint

Implemented the pre-registered level-1 gate in `ATTENTION.md` without editing
that binding file. Tiny deterministic checks confirm softmax weights sum to
one, the gate is exactly `196 * alpha * feature`, and trainable parameter counts
are QCNN-attn 138, B-PCNN-attn 138, and B-ATTN 24. All qnodes remain
`default.qubit` with `backprop`; both quantum pooling layers remain unchanged
and trainable. Python compilation passed for all changed/new scripts.

The A2 job launched at 2026-07-12 06:13 remained alive and CPU-active during
this checkpoint. In accordance with the machine-ownership precondition, no
Phase A3 smoke or training was launched. The 20-minute QCNN-attn smoke gate is
therefore pending, not passed.

## 2026-07-14 — Phase A2.3 trigger, pre-flight, and budget smoke (HOME PC)

The pre-registered A2.3 trigger evaluation is recorded verbatim:

> mean test macro-F1 L=1 = 0.4481, L=2 = 0.4992 (+0.0511 >= +0.02), L=3 = 0.4783, mag = 0.5442 (+0.0961 >= +0.02). Best A2.1 depth = L=2. Therefore A2.3 = reupload_L=2 with encoding amplitude10_mag, seeds 0,1,2.

Pre-flight inspection found that no new code was needed. qcnn_model.py
amplitude-embeds inputs[..., :1024], applies the four fixed magnitude rotations
to the designated readout wires 6–9, and leaves the L>1 re-upload layer on
inputs[..., :10]. Its qnodes remain default.qubit with backprop.
train_qcnn.py already accepts --encoding amplitude10_mag, --reupload-L, and
--tag, and writes configuration-tagged artifacts. Unit-level model construction
independently reported 173 trainable parameters for L=2 with amplitude10_mag.

On the HOME PC (WORKSPACE, NVIDIA GeForce RTX 5070), the only run
launched during implementation was this one-epoch budget smoke:

    & "WORKSPACE\.venv-e1\Scripts\python.exe" -u .\full_qcnn_pilot\scripts\train_qcnn.py --split-seed 0 --encoding amplitude10_mag --reupload-L 2 --tag smoke_L2_amplitude10_mag --batch-size 4 --lr 0.01 --smoke

The smoke reported exactly **173 parameters** and **218.7549383 seconds** for
the complete training-plus-validation epoch (3.65 minutes), with validation
Macro-F1 0.275739. It therefore passed the fixed 20-minute gate without any
downscaling or hyperparameter change. A before/after inventory confirmed that
no pre-existing result file's mtime changed. The uniquely tagged smoke
checkpoint, history CSV, and metrics JSON were deleted after verification.

A recursive pre-flight search found no abandoned partial L3 seed-2 artifact in
this checkout. The only matching files were the completed official seed-2
checkpoint, 88-epoch history CSV, and final metrics JSON, so none was deleted;
this preserves the official L2/L3 and Phase-A evidence as required.

The author-launch-only A2.3 block was appended to RUN_COMMANDS.md for seeds 0, 1,
and 2, sequential and never concurrent, with the fixed L=2, amplitude10_mag,
tag L2_amplitude10_mag, 100 epochs, patience 15, batch size 4, and learning rate
0.01. **No official run was launched during implementation.**

## 2026-07-14 — Phase A2.3 complete (L=2 + amplitude10_mag, home PC, 3 seeds)

The pre-registered conditional combined run finished on the home PC (RTX
5070, .venv-e1). All three seeds completed with the full test block and
params=173 (mag adds no trainable parameters over L=2 amplitude10):

| seed | best val macro-F1 | test macro-F1 | test acc |
|---|---|---|---|
| 0 | 0.6016 | 0.5239 | 0.6880 |
| 1 | 0.5570 | 0.5292 | 0.7131 |
| 2 | 0.6189 | 0.5861 | 0.7382 |

Mean test macro-F1 **0.5464 ± 0.0345**.

Finding (see results/function_class.md): depth and magnitude restoration do
NOT stack. A2.3 (0.5464) is statistically indistinguishable from A2.2
magnitude-alone at L=1 (0.5442 ± 0.0473); adding L=2 re-uploading on top of
the magnitude encoding buys nothing. Magnitude is the larger single effect
(+0.096 over the L=1 amplitude baseline 0.4481); depth is modest and
non-scaling (L=2 +0.051, L=3 +0.030). Every QCNN configuration remains well
below the B-QUAD quadratic null (0.7264) — best variant trails by 0.180.
The bottleneck is the 8-feature readout and quadratic ceiling, structural to
the amplitude-encoding + shallow-readout design, not representational depth.
This confirms the pre-registered expectations (A2.1 modest, A2.2 larger,
both still trailing B-QUAD) and closes the function-class story.

Deliverables written: results/ablations_all_runs.csv (15 rows: L1/L2/L3
amplitude + L1/L2 mag, 3 seeds each, with tag column); results/
function_class.md rewritten with the full mean±std table and verdict.
Runs were author-launched; no official run was launched during implementation.

## 2026-07-15 - Phase A3 verification and budget smoke (HOME PC)

The binding 2026-07-14 reconciliation in `ATTENTION.md` Section 2 was
confirmed before execution: A2.3 did not crown a new base, so Phase A3 remains
QCNN `amplitude10_mag`, L=1, with 133 base parameters. Inspection confirmed
the shared five-parameter scorer/gate, `alpha = softmax(score / 1)`,
`feature' = (196 * alpha) * feature`, unchanged trainable quantum
pool1 -> pool2 -> head, identical B-PCNN gating, and the 24-parameter B-ATTN
null. Independent construction reported parameter counts 138, 138, and 24.

The only run launched during implementation was the allowed one-epoch smoke:

    & "WORKSPACE\.venv-e1\Scripts\python.exe" -u .\full_qcnn_pilot\scripts\attention_experiment.py --arm qcnn_attn --split-seed 0 --smoke

The complete training-plus-validation epoch took **133.1297 seconds**, below
the fixed 1200-second stop gate, with validation Macro-F1 0.307831. The emitted
metrics asserted **138 trainable parameters**. No test evaluation occurred in
smoke mode.

An exact fingerprint of all pre-existing files under `results/` and
`checkpoints/` remained 60 files with SHA-256
`d4f0755eb70fd3b7bc65f70fe3e24f3a97a22f42401268133eb24d83cd059640`
after excluding and then deleting the three uniquely tagged smoke artifacts.
No pre-existing result mtime changed, and no smoke artifact remains.

The full three-arm by three-seed training protocol, three seed-0 saliency
passes, and finalizer are recorded in `RUN_COMMANDS.md` for manual execution. **No official run was launched during implementation.**

## 2026-07-16 — Phase A3 attention complete (home PC, 3 seeds + seed-0 saliency)

All three arms trained (author-launched, .venv-e1). Mean ± sample std test
Macro-F1 over seeds 0/1/2:

| Arm | Params | Test Macro-F1 |
|---|---|---|
| QCNN-attn | 138 | 0.5068 ± 0.0542 |
| B-PCNN-attn | 138 | 0.3793 ± 0.0910 |
| B-ATTN | 24 | 0.3493 ± 0.0642 |
| QCNN-mag (anchor) | 133 | 0.5442 ± 0.0473 |
| B-PCNN (anchor) | 133 | 0.2922 ± 0.0078 |
| B-QUAD (anchor) | n.a. | 0.7264 ± 0.0198 |

Verdicts vs ATTENTION.md §6 (see results/ATTENTION_REPORT.md):
- QCNN-attn vs QCNN-mag: -0.0374 — the ±0.03 "modest gain or tie" expectation
  did NOT hold; the gate slightly HURT the quantum model (within one seed's
  noise, but outside the pre-registered band). Reported as-is.
- QCNN-attn vs B-PCNN-attn: +0.1274 — QCNN side stayed ahead, as expected.
- B-ATTN below both gated hierarchical arms (-0.1575 vs QCNN-attn,
  -0.0300 vs B-PCNN-attn); no numeric "NEAR" threshold was pre-registered so
  none was invented. B-ATTN did not overtake the hierarchy.
- Level-2 quantum-attention trigger (≥ +0.02 gain): NOT met; stays parked.
- Saliency: reviewed; expectation partially borne out — leaf/lesion-concentrated
  for diseased classes (b7, b8), background/soil for healthy (00_0), diffuse
  overall. Honest mixed result; not a precise lesion detector, no such claim.

Headline: attention adds no accuracy on this task (the 5-param gate cannot
overcome the amplitude-encoding/readout bottleneck already established by the
A2 function-class study), but yields a defensible class-dependent
interpretability artifact. This closes the QCNN track. No official run was
launched during implementation.
