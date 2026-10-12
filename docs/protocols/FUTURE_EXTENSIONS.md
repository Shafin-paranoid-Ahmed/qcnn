# Full-QCNN — Parked Future Extensions

Status updated: 2026-09-07; original proposal text dated 2026-07-12.
**Level-1 patch attention is complete**, under the separate
[Phase A3 specification](../../ATTENTION.md), with
[results and saliency](../../results/ATTENTION_REPORT.md).
It did not improve on QCNN-mag, and the Level-2 trigger was not met. Other
extensions below remain parked unless separately authorized; their speculative
descriptions are retained as history. Base Phase A and A2 ablations are also
complete, so the old prerequisite to obtain a first clean result is satisfied.

**Honest framing for all of these:** each is architecturally legitimate and would
raise the architecture's novelty, but none is expected to produce a *quantum
advantage* — every one has a classical twin that will most likely tie under a fair
baseline (the same pattern already confirmed for the VQC head, quantum walk, and
quantum pooling). Add them for architectural richness / interpretability, not as a
route to a quantum win.

---

## 1. Attention over patches (most promising extension)

Not every leaf patch matters equally (lesion/edge vs background). Attention lets
the model learn which patches drive the prediction.

- **Best payoff (lead with this if implemented):** attention weights give a
  **saliency map over the leaf** — "the model attended to these lesion patches."
  A concrete, defensible interpretability figure that holds regardless of accuracy
  or any quantum advantage. Strong for an agricultural thesis.
- **Do it hierarchically/windowed, NOT global over 196 patches** (196² ≈ 38k
  pairs). Attend inside 2×2 / 4×4 windows → regional tokens → regional attention →
  global.
- **Anti-degeneracy:** if attention AND combination both go classical, the quantum
  part collapses to per-patch quanvolution (already exists). Keep the quantum
  pooling so novelty survives. Classical attention + quantum pooling is fine.
- **Implementation levels (do in order):**
  1. Classical attention over measured quantum patch features (cheap, trainable) —
     the first working version.
  2. Quantum query–key attention scores (swap-test / fidelity / joint circuit) —
     "more genuinely quantum," and the honest test is: does quantum attention beat
     classical attention on the *same* features? (Expected: ties.)
  3. Fully coherent quantum attention (entangle patch states before pooling) —
     hardest, big coherent state, Phase-3+ only.
- **Mandatory ablations if built:** patch-QCNN no-attention; + classical attn;
  + quantum attn; classical patch-CNN + classical attn; quadratic patch features
  + attn; attention-only no quantum. Key comparison: quantum attn vs classical
  attn on identical patch features.
- Prior art: QSANN / QSAN / quantum ViT (arXiv 2205.05625); quantum SE channel
  attention (arXiv 2507.11217).

## 2. Speculative global quantum attention (efficiency only)

Draft–verify idea borrowed from speculative decoding (Medusa, arXiv 2401.10774;
speculative sparse attention, arXiv 2606.30389 "Predict, Reuse, Repair"). Cheap
draft heads propose likely long-range patch links; an expensive quantum
query–key verifier checks only those → cost ~O(Nk) instead of O(N²).

- **Solves an efficiency/scalability problem ONLY.** Does not add accuracy,
  robustness, or quantum advantage. Its only honest claim: "speculative routing
  makes quantum global attention feasible for full-resolution patch-QCNNs."
- Prerequisite: you must FIRST have global quantum attention that is too expensive
  — i.e., extensions #1 level 2/3 working. This optimizes a bottleneck you have
  not hit yet. Lowest priority.
- If built, compare vs full global attention, windowed attention, top-k sparse
  attention, token pruning, and a fully classical draft–verify twin.

## 3. Channel / scale attention

- Channel attention: which of R/G/B or learned quantum observables matter
  (quantum squeeze-and-excitation, arXiv 2507.11217).
- Scale attention: combine fine patches (lesion texture) + medium regions (lesion
  groups) + large regions (spread). Potentially useful because plant severity
  depends on both lesion appearance and distribution.

## 4. Quantum combinatorial patch selection (QUBO/QAOA efficiency module)

Full proposal: `docs/still_exploring/combinatorial_vision/QUANTUM_COMBINATORIAL_VISION_DIRECTION.md`.
Use a quantum optimizer (QAOA / annealing) to solve a QUBO that selects the K most
informative of the 196 patches (relevance − redundancy − cardinality penalty)
BEFORE the patch-QCNN runs. Selected-patch maps double as an interpretability
figure; running the QCNN on K < 196 patches directly cuts the ~2.7 min/epoch cost.

- **Best-shaped quantum question so far:** discrete subset selection is QAOA's
  native problem class, unlike "make a circuit classify images." The research
  question (§12 of the proposal) is well-posed and its fairness rules (§7–§9:
  same scores, same budget, same classifier, only the optimizer changes) match
  this pilot's baseline discipline.
- **Why it is parked — three walls (assessed 2026-07-12):**
  1. **Not simulable at problem size.** QAOA needs one qubit per binary variable
     → 196 qubits → 2^196 amplitudes. Statevector ceiling in this environment is
     ~25 qubits (~0.5 GB); the QUBO's dense similarity couplings entangle
     everything, so no chunking/tensor-network escape. Options are: shrink to a
     ~5×5 patch grid (abandons the 224×224 identity), real hardware (D-Wave/IBM
     access mid-thesis), or "quantum-inspired" SA (the classical baseline in a
     costume).
  2. **Known outcome is a tie at best.** Low-depth QAOA does not beat simulated
     annealing on small/medium QUBOs; the relevance−redundancy objective is
     submodular-ish, so greedy is near-optimal and exact IP solves 196 variables
     in milliseconds. When classical is already optimal, "better subsets" is off
     the table.
  3. **Attacks a problem the pilot doesn't have.** Phase A showed the QCNN's
     bottleneck is the magnitude-discarding encoding + 8-feature readout, not
     too many patches. Selecting K < 196 patches removes information. The QUBO's
     relevance/similarity scores are classical feature engineering anyway.
- **Honest claim if ever built:** "a quantum combinatorial optimizer produced
  competitive sparse patch selection inside a real full-resolution pipeline" —
  an efficiency/interpretability module, NOT accuracy or quantum advantage.
- **Un-park trigger:** hardware access (or a ≤25-variable reformulation that is
  still meaningful) AND the base pilot fully written up. Mandatory baselines per
  the proposal: greedy top-K, k-medoids, submodular, SA, IP — same inputs, same
  budget.

## 5. IoT fusion

Entangle a raw-IoT qubit register with the image register in the later pooling
levels (multimodal quantum fusion). Separate research effort; out of pilot scope.

## 6. Fully coherent "Branch B" pooling

All-patches-entangled pooling with no intermediate measurement — where higher-order
expressivity AND the barren-plateau / classical-simulability tension live. Separate
multi-month effort; explicitly OUT of the pilot (see `full_qcnn.md` §12).

## 7. GPU training-speed engineering (kernel fusion / float32) — PARKED until the ladder closes

Added 2026-07-13, after the measured GPU verdicts. Background: the EBM sampler's
GPU port won 102-763x (one large 2^n evolution — ideal GPU shape) and was
adopted; the QCNN's torch-GPU twin measured only **1.45x at float64** on an
RTX 5070 (under the pre-registered 3x adoption bar; verdict "stay on CPU" in
`qcnn_gpu_bench/BENCH_LOG.md`). Root cause is workload shape, not GPU muscle:
196 tiny 10-qubit circuits per image launch thousands of small kernels, and
per-kernel launch overhead — identical on any card, including the lab 4090 —
dominates. A bigger GPU does not change this; only software does.

Two known levers, both real engineering with re-validation attached:
1. **Kernel fusion / batching** — restructure so the 196 patch circuits (x batch)
   execute as few large batched tensor ops instead of many small ones.
2. **float32 path** — measured ~4x in the same benchmark, but numerically
   different; adoption-ineligible without its own equivalence study.

Un-parking conditions (ALL required):
1. The pre-registered ladder (A2.1 L=3, A2.3, Phase A3 attention) is COMPLETE —
   never a mid-ladder backend/precision switch (anchor-comparability rule).
2. A concrete consumer for the speed is green-lit: the full five-class task at
   proper budget and/or Phase B corruption robustness. Speed with no consumer
   is engineering for its own sake — leave parked.
3. Stage 2 of the EBM track (the invention headline) is not starved of team
   runway by this work — priority order is Stage-1 matrix -> Stage 2 -> ladder
   completion -> this.
4. Validation gate before any new science runs on the new path: retrain one
   known config (e.g. mag L=1 seed 0) and reproduce its result within seed
   noise; new tag for everything downstream.

---

## Decision rule for un-parking any of these

Un-park an idea ONLY if all hold:
1. The base patch-QCNN (Phase A) has cleared its go/no-go with a real number.
2. There is enough thesis runway for another research-grade component.
3. The idea adds either (a) a defensible architectural-novelty claim, or (b) a
   concrete interpretability artifact — not merely "more quantum."

Otherwise leave it here. A finished thesis on the base architecture beats an
unfinished one with five half-built mechanisms.
