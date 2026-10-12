> Historical thesis record. Original machine paths and commands are retained as provenance. Use the standalone [README](README.md) and runbook for current setup.

# Full-QCNN Pilot Change Log

Updated whenever this folder changes.

## 2026-07-12

- Wrote `ATTENTION.md`: pre-registered Phase-A3 spec (FUTURE_EXTENSIONS §1
  level 1 un-parked for the parallel-track plan). Classical 5-param attention
  gate over quantum patch features → saliency maps as the headline deliverable;
  arms QCNN-attn (138 params), B-PCNN-attn (138, param match restored), B-ATTN
  null; anchors reused. Level 2 (quantum attention scores) conditional on
  ≥ +0.02 gain; levels 2–3 otherwise parked. Runs on the lab PC; the EBM pilot
  (`docs/still_exploring/quantum_sampler_ebm/QSAMPLER_EBM_PREREG.md`) runs on
  the local machine — one track per machine, git-only sync.
- Wrote `ABLATIONS.md`: pre-registered Phase-A2 ablation spec for the
  implementing agent. A2.1 = re-uploading depth L∈{2,3} (function-class /
  scalability; params grow to 173/213, so param-match holds only at L=1);
  A2.2 = `amplitude10_mag` magnitude-preserving encoding (finishes the branch
  already present in `patch_encode.py`; 133 params preserved, existing
  baselines remain valid); A2.3 = conditional combined run. Step 0 mandates a
  `--tag` on `train_qcnn.py` outputs because untagged ablation runs would
  overwrite the frozen Phase-A checkpoints/histories.

## 2026-07-11

- Created `full_qcnn_pilot/` scaffold from the spec
  `docs/still_exploring/full_qcnn/full_qcnn.md`.
- Verified data layout: images at `Data/test/<id>/<id>.jpg` (all 5,767 ids
  resolve; folders also hold IoT `<id>.csv/.json/_final.csv`). Corrected the spec
  image-path (it previously said `Data/train/...`).
- Wrote `HANDOFF.md` (build order, constraints, file status), `README.md`,
  `RUN_COMMANDS.md`, `RESULTS_LOG.md`, this changelog.
- Implemented `scripts/build_splits.py` (DONE): true-label splits from
  `Data/test.csv`, tasks {crop3_disease, healthy, species, stage}, stratified
  60/20/20, 3 seeds, JSON output.
- Implemented `scripts/phase0_resolution.py` (DONE): classical Macro-F1 vs
  resolution sanity, no quantum deps.
- Added stubs (implement per spec): `patch_encode.py` (§5), `qcnn_model.py` (§6),
  `train_qcnn.py` (§7), `baselines.py` (§8), `metrics.py` (§10),
  `run_experiment.py` (§10-11), `corrupt_eval.py` (§9, Phase B only).
- Implemented the Phase-A raw-pixel path in `patch_encode.py`: per-image fixed
  normalization, 16x16 patch extraction, RGB Morton ordering, and 10-qubit
  amplitude-vector preparation. Implemented the base `qcnn_model.py`: shared
  patch circuit, measured Level-1 and Level-2 trainable quantum pooling, and a
  linear readout. No learned classical encoder is present.
- Ran the required early simulation-cost check. The primary 10-qubit path is
  functionally correct but exceeds the Section 7 budget for the five-class task;
  fallback encoding must be implemented before a training smoke or full run.
- Added the Section 5.3 `channel8` fallback: three 8-qubit Morton-ordered RGB
  registers processed by the same patch circuit and combined before quantum
  pooling. Benchmarking showed it is slower than the primary path, so it does
  not clear the training-budget gate.
- Implemented `metrics.py` and `train_qcnn.py`, including class-weighted clean
  training, validation-only selection, unreweighted probability metrics,
  gradient-norm logging, checkpoints, and the one-epoch smoke mode.
- Implemented `baselines.py` with the exact 133-parameter B-PCNN hierarchy and
  explicit degree-2 B-QUAD control. Implemented `run_experiment.py` for resumable
  seed-wise Phase-A execution and aggregate CSV/Markdown outputs.
- Audited calibration fairness: added a second unweighted B-QUAD probability
  model for ECE/Brier/NLL and explicitly recorded genuine-softmax calibration
  sources for QCNN and B-PCNN. Confirmed the `default.qubit` + `backprop` smoke
  at 156.3 seconds per complete epoch.
- Completed clean Phase A across seeds 0, 1, and 2 for QCNN, exact-matched
  B-PCNN, and B-QUAD. Generated all aggregate tables/reports and recorded the
  clean competitiveness pass plus the decisive B-QUAD result.
- Corrected B-PCNN after review: replaced pre-averaged channel-mean inputs with
  a full raw-spatial patch filter while preserving exactly 133 parameters and
  the 4-feature/8-feature hierarchy; matched QCNN optimization and added three
  B-PCNN histories/checkpoints. Reran all B-PCNN/B-QUAD seeds and regenerated
  aggregate artifacts without retraining the valid QCNN seeds.
- Switched `scripts/qcnn_model.py` qnodes from `lightning.qubit`/`adjoint` to
  `default.qubit`/`backprop` to broadcast the 196 per-patch circuits (46x faster
  patch layer; full model ~2.7 min/epoch vs ~48 min). Unblocks Phase-A training.
  Updated spec Sections 2 and 7 accordingly.

## 2026-07-12

- Added `ABLATIONS.md` for the pre-registered Phase A2 studies and recorded the
  authoritative §5.5 reconciliation: `amplitude10_mag` injects fixed rotations
  on readout wires 6–9; its expectations remain unchanged.
- Added `--tag` to `scripts/train_qcnn.py`, defaulting to
  `L{reupload_L}_{encoding}`, so new checkpoints, histories, and metrics use
  configuration-specific names without modifying frozen Phase-A results.
- Completed only the A2.1 L=2 and L=3 seed-0 budget smokes. Both passed the
  20-minute gate (149.9 and 140.2 seconds/epoch respectively). No full A2.1 or
  conditional A2.3 run was launched by the agent.
- Confirmed the existing three wires-6–9 magnitude seeds as official A2.2
  results: mean test Macro-F1 0.544200 +/- 0.047340. A2.3 remains pending the
  manually launched A2.1 full-run results.
- Implemented Phase A3 level-1 attention without changing `ATTENTION.md`:
  a default-off shared linear patch scorer and mean-preserving gate in the
  QCNN, the identical gate in B-PCNN, and the B-ATTN channel-mean null.
  Assertions enforce 138 parameters for QCNN-attn and B-PCNN-attn; B-ATTN has
  24 parameters for this five-class task.
- Added `scripts/attention_experiment.py` for uniquely tagged Phase A3
  training and seed-0 saliency CSV/PNG export, plus
  `scripts/finalize_attention.py` for `attention_runs.csv` and the final report.
  No run was launched while the user-owned A2 Python process remained active.

## 2026-07-14 — hardware-move reconciliation: qcnn_L3_amplitude10_seed2

- A2.1 L=3 seed 2 was completed on the research PC (RTX 5070, G:\FOR
  THESIS\MAIN), not the lab PC that ran seeds 0-1 (its metrics.json history
  path records the G: drive). Same code, same tagged naming, identical
  hyperparameters; merged under the Step-0 rule for cross-machine results.
  Test macro-F1 0.4664. Any partial L3 seed-2 remnants on the lab PC are to
  be deleted before the next sync (instruction included in the A2.3 handoff).
- A2.3 trigger evaluated per ABLATIONS.md §A2.3: L=1 mean test macro-F1
  0.4481; best A2.1 = L=2 at 0.4992 (+0.0511 >= +0.02); A2.2 mag 0.5442
  (+0.0961 >= +0.02). Both conditions met — A2.3 FIRES as L=2 +
  amplitude10_mag, seeds 0-2, handed to the lab-PC agent.

## 2026-07-14 — Phase A2.3 HOME-PC pre-flight and handoff

- Verified by inspection that the existing L=2 amplitude10_mag path matches the
  reconciled pre-registration: raw-pixel amplitudes, fixed magnitude rotations
  on wires 6–9, unchanged first-10-value re-uploading, default.qubit + backprop,
  and tagged outputs. No code change was required.
- Independently constructed the model and confirmed 173 trainable parameters.
  The permitted one-epoch smoke then reported 173 parameters and 218.7549383
  seconds/epoch on the HOME PC (NVIDIA GeForce RTX 5070), passing the fixed
  20-minute gate. No pre-existing result mtime changed, and the three tagged
  smoke artifacts were deleted after verification.
- Searched recursively for abandoned partial L3 seed-2 remnants. None existed;
  the three matching files form the completed official seed-2 bundle and were
  preserved with all other official L2/L3 and Phase-A artifacts.
- Appended the sequential, user-launch-only A2.3 seed 0–2 command block to
  RUN_COMMANDS.md. **No official run was launched by the agent.**

## 2026-07-15 - Phase A3 verification and smoke gate

- Reconciled the binding base against `ATTENTION.md` Section 2: A2.3 did not
  trigger a base switch, so QCNN-attn remains `amplitude10_mag`, L=1, with
  133 base parameters plus the five-parameter shared scorer/gate.
- Verified the three arms, mean-preserving `196 * alpha * feature` gate,
  fixed temperature 1 softmax, gate placement, trainable quantum pooling,
  Phase-A optimizer/scheduler/loss/batch/patience/epoch protocol, seed-0
  saliency CSV/PNG paths, and tagged artifact names against Sections 3-5.
  Independent construction reported 138 QCNN-attn parameters, 138
  B-PCNN-attn parameters, and 24 B-ATTN parameters.
- Ran only the permitted qcnn-attn seed-0 one-epoch smoke. It completed in
  133.1297 seconds with validation Macro-F1 0.307831 and asserted 138
  parameters, passing the fixed 1200-second gate without downscaling.
- Proved artifact isolation with an exact pre-existing results/checkpoints
  fingerprint: 60 files and SHA-256
  `d4f0755eb70fd3b7bc65f70fe3e24f3a97a22f42401268133eb24d83cd059640`
  both before and after cleanup. Deleted the three uniquely tagged smoke
  artifacts; no pre-existing result mtime changed.
- Corrected `finalize_attention.py` so `attention_runs.csv` uses every
  `all_runs.csv` column plus `tag`, and copies the nine frozen QCNN-mag
  L=1, B-PCNN, and B-QUAD anchor rows unchanged apart from the reuse tag.
  Added the required run metadata and the pre-run `ATTENTION_REPORT.md`
  scaffold. The generated verdict now covers QCNN-attn versus B-PCNN-attn,
  B-ATTN versus both gated arms without inventing a numeric "NEAR" threshold,
  an explicit human-review slot for the pre-registered B-ATTN "NEAR"
  conclusion, and an explicit qualitative saliency-review slot.
- Appended the complete user-launch-only Phase-A3 protocol to
  `RUN_COMMANDS.md`: three arms by seeds 0/1/2, all three seed-0 saliency
  passes, then finalization. Also recorded the exact completed HOME-PC smoke
  command separately as historical evidence. **No official run was launched
  by the agent.**

## 2026-09-07 — Current documentation restored

- Replaced stale README/handoff instructions that described implemented
  quantum scripts as stubs or prescribed the superseded adjoint simulator.
- Recorded completed Phase A/A2/A3 state and linked the final reports;
  Phase B remains a stub and Level-2 attention remains untriggered.
- Marked old run commands and preregistration execution instructions as history.
- Corrected duplicated per-class columns in the Markdown clean report without
  changing the underlying values, CSVs, code, checkpoints, or expectations.
- No training, evaluation, environment change, or new test run.
