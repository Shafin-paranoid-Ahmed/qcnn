# Phase A3 — Patch Attention + Saliency (pre-registered implementation spec)

> **Completion update — 2026-09-07:** Phase A3 finished on the home PC on
> July 16; the original lab-PC assignment below is historical. QCNN-attn
> Macro-F1 is 0.5068 ± 0.0542, below QCNN-mag 0.5442 ± 0.0473. The Level-2
> trigger was not met. Seed-0 saliency was reviewed with mixed qualitative
> findings; the B-ATTN "NEAR" verdict remains pending human interpretation.
> See [the attention report](results/ATTENTION_REPORT.md). The original
> registered expectations below are preserved and are not pending run tasks.

Date: 2026-07-12 (written before any attention code exists; expectations in §6
must never be edited after results exist)
Runs on: **lab PC** (`WORKSPACE`, `.venv-e1` interpreter). This track
owns `full_qcnn_pilot/` on that machine; the EBM pilot runs on the other machine.
One track = one machine = one folder; sync via git only.
Depends on: Phase A (frozen) and the A2 ablations (`ABLATIONS.md`). Read
`HANDOFF.md` first — all its hard constraints still bind (true labels only, raw
pixels in the quantum path, `default.qubit` + `backprop`, never touch `docs/` or
any `.docx`/`.pdf`).
Origin: FUTURE_EXTENSIONS.md §1, **level 1 only** un-parked (2026-07-12 decision:
team runs tracks in parallel). Levels 2–3 remain parked except the conditional in
§7. Extensions #2, #4, #5, #6 remain fully parked.

## 1. Purpose (honest framing)

Attention over patches is implemented for ONE headline payoff: a **saliency map
over the leaf** — "the model attended to these lesion patches" — a defensible
interpretability artifact for an agricultural project that holds regardless of
accuracy. Accuracy gain is NOT expected to be large and is not the claim.
No quantum-advantage claim under any outcome.

## 2. Base configuration (pre-registered)

- Base model: **QCNN with `amplitude10_mag` encoding, L=1** (the current best
  variant; 133 params). If the A2 ladder's final verdict crowns a different
  config via the A2.3 trigger, the base may be switched ONCE, with a dated
  reconciliation note added here BEFORE any attention run — never after.

  > **2026-07-14 reconciliation (before any attention run): base stays L=1
  > mag — A2.3 does NOT crown a new config.** A2.3 (L=2 + amplitude10_mag)
  > completed at mean test Macro-F1 0.5464 ± 0.0345, statistically
  > indistinguishable from A2.2 magnitude-alone at L=1 (0.5442 ± 0.0473);
  > depth and magnitude do not stack (see results/function_class.md). The
  > simpler L=1 mag config (133 params vs 173) remains the best variant, so
  > the pre-registered base is unchanged. No base switch.
- Splits: seeds 0/1/2 from `full_qcnn_pilot/splits/`. Training protocol
  unchanged (Adam lr 1e-2, cosine, class-weighted CE, batch 4, patience 15, max
  100 epochs, val-only selection, test once per seed).
- Tagged filenames per `ABLATIONS.md` Step 0 (e.g. `qcnn_attn_L1_mag_seed0_*`).
  Never overwrite Phase-A or A2 artifacts.

## 3. Level-1 design: classical attention gate over quantum patch features

Smallest thing that produces a saliency map, added AFTER the patch layer and
BEFORE Level-1 quantum pooling:

- Patch layer outputs `f_i ∈ R^4` for each of the 196 patches (existing).
- Scorer: a single shared linear map `s_i = w·f_i + b` (**5 trainable params**,
  no hidden layer). Attention weights `α = softmax(s / τ)` over the 196 patches,
  temperature τ = 1 fixed.
- Gating: `f_i' = (196 · α_i) · f_i` (scale factor 196 keeps the mean feature
  magnitude unchanged so pooling-layer input statistics are preserved).
- The gated features flow into the UNCHANGED quantum pool1 → pool2 → linear head.
  Quantum pooling stays trainable — this is the anti-degeneracy rule: classical
  attention + quantum pooling is allowed; attention replacing quantum pooling is
  not.
- Total params: 133 + 5 = **138. Assert it.**
- Saliency map = α reshaped to the 14×14 patch grid.

## 4. Arms (3 seeds each; anchors reused, never retrained)

| Arm | What | Params | Status |
| --- | --- | ---: | --- |
| QCNN-attn | §3 on the §2 base | 138 | NEW — train |
| B-PCNN-attn | identical 5-param scorer+gate on B-PCNN's 4-d patch features, before its cell-mean pooling | 138 | NEW — train (param match restored: 138 vs 138) |
| B-ATTN | attention-only null: same scorer/gate over each patch's 3 channel means → mean-pool → linear head (no quantum, no hierarchy) | ~25 | NEW — train (cheap; isolates "the attention did the work") |
| QCNN-mag L=1 | no-attention anchor | 133 | reuse A2.2 |
| B-PCNN, B-QUAD | Phase-A anchors | 133 / n.a. | reuse |

Key comparisons: QCNN-attn vs QCNN-mag (does the gate help the quantum model?);
QCNN-attn vs B-PCNN-attn (param-matched, both gated); B-ATTN vs everything
(is attention alone doing the work?).

## 5. Deliverables

1. Tagged histories/checkpoints/metrics for the three new arms.
2. `results/attention_runs.csv` — per (arm, seed), columns as `all_runs.csv`
   plus `tag`; anchor rows copied in for side-by-side reading.
3. **Saliency artifacts** (the headline): for each seed-0 test class, the 3
   highest-confidence correct and 2 misclassified images — per image, the 14×14
   α grid as CSV in `results/saliency/` AND a heatmap overlay PNG in
   `results/figures/`. (Dated amendment to the folder rule: `results/figures/`
   MAY contain PNG — figures are legitimate pilot outputs; the MD/PY/CSV/JSON
   rule otherwise stands.)
4. `results/ATTENTION_REPORT.md`: the arm table (mean ± sample std over 3
   seeds: Macro-F1, accuracy, ECE, Brier, NLL, macro recall), the saliency
   figure index, and the honest verdict against §6.
5. Dated `RESULTS_LOG.md` + `CHANGELOG.md` entries (commands, wall-clock/epoch,
   hardware, interpretation).

## 6. Pre-registered expectations (do not edit after results exist)

- QCNN-attn vs QCNN-mag: **modest gain or a tie** (within ±0.03 Macro-F1 is the
  modal expectation). The gate helps only if uninformative background patches
  are currently diluting the pooled signal.
- QCNN-attn vs B-PCNN-attn: QCNN side expected to stay ahead (Phase-A pattern),
  but the gate may help B-PCNN more (its mean-pooling dilutes hardest).
- B-ATTN: expected well below both (attention over color means alone is a weak
  classifier) — if B-ATTN lands NEAR QCNN-attn, the honest conclusion is that
  attention, not the quantum hierarchy, carries the gain. Report it.
- Saliency: expected to concentrate on lesion/leaf regions rather than
  background for diseased classes (qualitative, figure-based claim only).
- Either direction on every row is reportable. No tuning to chase a number.

## 7. Level 2 — quantum attention scores (CONDITIONAL, default OFF)

Un-parked ONLY if: QCNN-attn improves on QCNN-mag by ≥ +0.02 mean test Macro-F1
AND the team confirms runway after the EBM pilot's G1/G2 gates. Then: replace
the linear scorer with a quantum query–key score (fidelity/swap-test between
patch feature states) and test **quantum attention vs classical attention on
identical features**. Pre-registered expectation: **ties**. Level 3 (coherent
attention) stays parked entirely.

## 8. Hard don'ts

- No changes to circuits, encodings, or training hyperparameters beyond the
  5-param scorer/gate. No new baselines beyond §4. No Phase B/corruption.
- No global 196×196 pairwise attention (that is extension #2's territory).
- No test-set selection; no editing §6 after results exist; no un-parking
  level 2 without the §7 trigger, recorded in a dated note.
- If a result contradicts an expectation, that IS the finding — report it.
