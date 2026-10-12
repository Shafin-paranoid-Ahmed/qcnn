# Phase A2 — Pre-Registered Ablations (spec for implementing agent)

> **Completion update — 2026-09-07:** A2.1 depth 2/3, A2.2 magnitude, and
> the triggered A2.3 combined run are complete. Three-seed test Macro-F1:
> L2 0.4992, L3 0.4783, magnitude 0.5442, combined 0.5464. All trail
> B-QUAD 0.7264. Read [function-class results](results/function_class.md).
> The original expectations and implementation-detail correction below remain
> historical preregistration; the listed runs are not pending work.

> **2026-07-12 implementation-detail reconciliation:**
> `docs/still_exploring/full_qcnn/full_qcnn.md` §5.5, dated 2026-07-11, is the
> authoritative pre-registration for `amplitude10_mag` and specifies injecting
> the four fixed scalars with `RY` on the designated readout wires 6–9. The
> later A2.2 wording below that says wires 0–3 was a divergent cross-machine
> draft. For this pilot, wires 6–9 override that implementation detail. The
> pre-registered expectations below are unchanged. Existing A2.2 seeds using
> wires 6–9 are conforming official runs and must not be rewired or rerun.

Date: 2026-07-12
Depends on: completed clean Phase A (see `RESULTS_LOG.md` final verdict,
`results/QCNN_REPORT.md`). Read `HANDOFF.md` first; all its hard constraints
still apply (true labels only, raw pixels in the quantum path, `default.qubit`
+ `backprop`, only MD/PY/CSV/JSON inside `full_qcnn_pilot/`, never touch
`docs/` or any `.docx`/`.pdf`).

## Why these two ablations

Phase A result: QCNN 0.448 macro-F1 beats param-matched B-PCNN (0.292) but
trails the quadratic null B-QUAD (0.726) by 27.8 points. Diagnosis: (1) the
amplitude encoding L2-normalizes each patch, discarding the absolute
color/brightness signal B-QUAD classifies on; (2) at L=1 the circuit is
~quadratic in pixels, same function class as B-QUAD but on worse inputs.
These ablations test each cause separately. They complete the function-class
story; they are NOT an accuracy chase.

## Pre-registered expectations (written BEFORE running — do not edit)

- **A2.1 (L=2, L=3):** modest gain over L=1's 0.448 at best; does NOT close
  the gap to B-QUAD, because re-uploading never restores the discarded
  magnitude. Note the implementation limitation: the re-upload layer injects
  only the FIRST 10 Morton-ordered normalized values per patch
  (`qcnn_model.py` `patch_qnode`), a thin signal. Keep it as-is (it is what
  the spec defined and what any lab-PC run used); state the limitation in
  interpretation.
- **A2.2 (magnitude encoding):** larger gain than A2.1, because it restores
  exactly the signal B-QUAD exploits; still expected to trail B-QUAD (the
  8-feature readout bottleneck and quadratic ceiling remain).
- Either outcome is reportable. Do not tune anything to chase a number.

## Step 0 — REQUIRED before any run: tag output filenames

`train_qcnn.py` currently writes `checkpoints/qcnn_seed{N}.pt`,
`results/qcnn_seed{N}_history.csv`, `results/qcnn_seed{N}_metrics.json` with
no config identifier. **Any ablation run overwrites the Phase-A artifacts.**

Add a `--tag` argument (default `"L1"` for backward naming clarity is NOT
allowed — default must be derived: `f"L{args.reupload_L}_{args.encoding}"`)
and write all three outputs as `qcnn_{tag}_seed{N}...`. Do NOT rename or move
the existing Phase-A files; they are the frozen record. Verify by running the
smoke and confirming no pre-existing file's mtime changes.

If results arrive from another machine (lab PC), merge them ONLY under tagged
names and record the hardware in the log entry; never overwrite the Phase-A
files.

## A2.1 — Data re-uploading depth (function-class / scalability study)

- Runs: `reupload_L ∈ {2, 3}` × seeds {0, 1, 2}, encoding `amplitude10`,
  `d_conv=2`, all other hyperparameters IDENTICAL to Phase A (batch 4,
  Adam lr 1e-2, cosine, patience 15, class-weighted CE, val-selected
  checkpoint, test touched once).
- **Parameter caveat (must appear in interpretation):** patch-layer weights
  scale as `(L, d_conv, 10, 2)` → total params 173 at L=2, 213 at L=3. The
  133-param match to B-PCNN holds only at L=1. A2.1 is therefore a
  *scaling/function-class* study, not a param-matched comparison. Record
  actual `params` per run (already a column in `scalability.csv`).
- Budget gate: run a 1-epoch `--smoke` at L=3 first; if the epoch exceeds
  20 min, STOP and report — do not downscale.

## A2.2 — Magnitude-preserving encoding `amplitude10_mag`

`patch_encode.py` already implements the data side (`amplitude10_mag`
branch): returns `[196, 1028]` = 1024 Morton/L2-normalized amplitudes ++
3 per-channel means ++ 1 scaled patch norm, all four scalars in `[0, π]`
(inputs come from `per_image_angle_normalize`, so means are valid angles).

Remaining implementation:

1. `qcnn_model.py`: accept `encoding="amplitude10_mag"`. Expected input
   shape `(196, 1028)`. In `patch_qnode`: `AmplitudeEmbedding` on
   `inputs[..., :1024]` as now, then inject the 4 magnitude scalars as
   input-dependent rotations — `qml.RY(inputs[..., 1024 + k], wires=k)` for
   k in 0..3 — BEFORE the variational layers. No new trainable parameters:
   the count must remain exactly 133 at L=1 (assert it). The re-upload layer
   at L>1 keeps using `inputs[..., :10]` unchanged.
2. `train_qcnn.py`: add `amplitude10_mag` to the `--encoding` choices.
3. This stays inside the "raw pixels into the quantum path" constraint:
   channel means and the patch norm are fixed arithmetic on raw pixels, not
   learned or engineered features (same status as the L2 normalization
   itself). State this in the log entry.
- Runs: L=1, `amplitude10_mag`, seeds {0, 1, 2}, all else identical to
  Phase A. Because params stay 133, the existing B-PCNN and B-QUAD numbers
  remain the valid comparators — do NOT retrain baselines.

## A2.3 — Combined run (CONDITIONAL — pre-registered trigger)

Run L=best(A2.1) with `amplitude10_mag` (seeds 0–2) ONLY IF both A2.1's best
L and A2.2 each improve mean test macro-F1 over the Phase-A L=1 value
(0.4481) by ≥ +0.02. Otherwise skip and say so.

## Deliverables

1. Tagged per-run histories/checkpoints/metrics (Step-0 naming).
2. `results/ablations_all_runs.csv` — one row per (config, seed) including
   Phase-A L=1 rows copied in for side-by-side reading; columns as in
   `all_runs.csv` plus `tag`.
3. Extend `results/scalability.csv` with the L=2/L=3 rows (params column
   filled).
4. Rewrite `results/function_class.md`: table of mean±std macro-F1 for
   L∈{1,2,3} and mag-encoding vs the fixed B-QUAD 0.7264 line; state plainly
   whether re-uploading and/or magnitude restoration moved the model beyond
   the quadratic null, referencing the pre-registered expectations above.
5. Dated entries in `RESULTS_LOG.md` and `CHANGELOG.md`: commands, wall-clock
   per epoch, hardware (this machine vs lab PC), honest interpretation.

## Hard don'ts

- No hyperparameter changes beyond `reupload_L` / `--encoding` / `--tag`.
- No baseline retraining, no new baselines, no attention, no Phase B.
- No test-set peeking; selection on val only, test once per run.
- No editing pre-registered expectations after results exist.
- If a result contradicts the expectation, that IS the finding — report it.
