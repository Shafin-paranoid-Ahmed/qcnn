# Full-Resolution Patch-QCNN — Implementation Spec (Pilot)

Last updated: 2026-07-11
Status: implementation brief. Read the WHOLE file
before writing any code. This document is the single source of truth for this
experiment. If anything here conflicts with older notes, THIS file wins.

Folder for all code/results of this experiment: `full_qcnn_pilot/`
(create it at the repository root, i.e. `WORKSPACE/full_qcnn_pilot/`).
Do NOT put code inside `docs/`.

---

## 0. What you are building, in one paragraph

You are building the **first full-resolution (224×224) hierarchical patch-based
Quantum Convolutional Neural Network (Patch-QCNN)** for crop disease-**stage**
classification, and evaluating it under image corruption against strong, fairly
matched classical baselines. The image is divided into a fixed grid of raw
patches (no learned classical encoder), each patch is quantum-encoded, a
**shared** quantum convolution circuit processes every patch (weight-sharing,
like a CNN filter), measured patch features are pooled hierarchically across
spatial scales, and a linear readout predicts the disease stage. The pilot runs
on ONE ordinal chain (`3_a9_1 → 3_a9_2 → 3_a9_3`, a 3-class head) so it is
tractable to simulate, while keeping the architectural novelty (full resolution,
direct pixel→quantum learning) 100% intact.

**The contribution this project is claiming is the ARCHITECTURE and its rigorous
characterization — not a quantum advantage.** A corruption-robustness or
calibration edge that survives the fair baselines is *upside*, not the
load-bearing claim. Do not tune, cherry-pick, or scale circuits in search of a
win; report what the architecture actually does.

The scientific question:

```text
Can a direct, hierarchical, patch-based QCNN that learns from raw 224×224 pixels
classify a real crop-image task at least as well as strong classical models
receiving the same raw images and identical training — and does it do anything
beyond the quadratic-in-pixels function class that amplitude-encoded quantum
models are known to collapse to?
```

### 0.1 Task target is CONFIGURABLE — pick the claim first

The novelty is the **architecture** (full-res pixels → hierarchical quantum conv),
which is independent of the label. Choose the target with `--task`:

- `crop3_disease` (RECOMMENDED default): one crop, several visually distinct
  conditions → forces colour/texture learning, not leaf-outline shortcuts. Use
  **crop 3, 5 classes**: `3_00_0` (healthy, 1177), `3_b3_1` (166), `3_b6_1` (142),
  `3_b7_1` (156), `3_b8_1` (153). In-domain, well-populated, avoids the
  stage-1-vs-2 ambiguity.
- `healthy` : binary healthy-vs-diseased (`disease == "00"` vs not). Easy, clean.
- `species` : predict `crop` (6 classes). Easiest to fit BUT scientifically
  weakest — a species classifier can cheat on leaf shape / background rather than
  learning disease-relevant colour and texture. Use only as a sanity check, not
  the headline task.
- `stage` : the original ordinal `3_a9` 3-class chain (Section 3.2). Hardest;
  keep only if the disease-**staging** claim specifically matters to the project.

If you pick a non-`stage` task, the class list, counts, and stratification in
Section 3.2 change accordingly; everything else in Section 3 (parsing, splits,
seeds, image-path resolution) is unchanged.

**Corruption robustness is an OPTIONAL axis, not mandatory (Section 9).** It is
the ONLY axis on which the QCNN can show a *difference* from classical (on plain
clean accuracy, strong classical models win or tie — that is the most certain
fact here). Decide up front which claim the project is making:

- Claim = "novel quantum vision architecture" → run clean-only; corruption off.
  Simplest build; contribution is the architecture + clean characterization.
- Claim = "quantum for corruption robustness" (the phase-2 pivot) → keep corruption;
  it is the differentiator that produces a quantum-flavored story.

Whichever task/claim you pick, the architecture, encoding, circuit, baselines,
and go/no-go structure below are UNCHANGED — only the label and whether the
corruption axis runs differ.

### 0.2 Recommended execution order — Phase A then Phase B (gated)

Run in this order; do not start Phase B until Phase A clears its gate.

**Phase A — prove the architecture (headline contribution).** Clean classification
only. Deliver: the trained Patch-QCNN, the clean fair-baseline table (QCNN vs
B-PCNN / B-QUAD / B-E0 etc.), the Phase-0 resolution curve (Section 4), the
scalability/ablation study, and the function-class study (Section 10.1). These
studies ARE the positive contribution and are independent of corruption. Gate:
proceed to Phase B only if the QCNN is *competitive* clean (go/no-go criterion 2,
Section 11) — otherwise stop and write up Phase A alone.

**Phase B — corruption characterization (secondary, honest due diligence).** Turn
on the corruption axis (Section 9) and add B-CT. Report the drop table. **A null
or fall-short corruption result is an EXPECTED and ACCEPTABLE outcome** — a
classical CNN with global pooling captures the same "global-feature survives local
corruption" effect, and the corruption-trained B-CT baseline will likely erase any
QCNN edge. Because the architecture (Phase A) is the headline, a null Phase B is a
reported property, not a failed claim. Do NOT retrofit robustness tricks into
Phase A to force a Phase B win, and do NOT pre-write the Phase B conclusion.

Corruption robustness must NOT be framed as the project headline. If Phase B shows
a narrow edge that survives all three seeds, B-QUAD/B-POLY2, and B-CT, that is
upside worth reporting — but the project stands on Phase A regardless.

### 0.3 Novelty positioning and claim wording — READ, this governs the writeup

**The novelty is the ARCHITECTURE, not the plant task.** Hybrid quantum-classical
QCNNs have already been applied to plant / apple-leaf disease classification, so
"quantum ML for plant disease" is NOT a novel claim. The defensible novelty is
the *full-resolution hierarchical patch processing*:

- preserving the full 224×224 input (prior direct-QCNN work centres on ~28×28
  MNIST or tiny synthetic RGB);
- encoding raw patches with NO learned CNN/ViT/PCA encoder in front;
- one shared quantum convolution reused across all patches;
- combining local patch information through a genuine spatial hierarchy;
- characterizing how it scales with patch count, qubits, and depth (Section 10.1).

**Do NOT claim "first full-quantum 224×224 image classifier."** Measuring patch
observables and re-encoding between pooling levels makes this a *hybrid* quantum-
classical network, not a coherently-quantum pipeline. Defensible claim wording:

> A full-resolution hierarchical patch-QCNN for real-world plant-leaf image
> classification, using raw-pixel quantum patch encoding without a learned
> classical image encoder, trained end-to-end.

"End-to-end trainable" = the patch circuits, pooling circuits, and linear head are
optimized jointly. It does NOT assert every stage is physically quantum.

**Anti-degeneracy requirement.** If the design collapses to "apply one shared
patch circuit independently to every patch, then average/concatenate the
measurements and feed a classical head," it is just **quanvolutional feature
extraction, which already exists** — not a new hierarchical QCNN. The pilot MUST
retain at least one genuine hierarchical quantum component — here, the trainable
Level-1 and Level-2 quantum pooling circuits that receive patch measurements and
re-process them (Section 6.2), plus data re-uploading across levels (Section 5.4).
Do not silently simplify these away to save time; if you must cut cost, cut depth
or patch count, never the hierarchy itself.

---

## 1. HARD CONSTRAINTS — read first, do not violate

Non-negotiable. Getting any of these wrong invalidates the experiment.

### 1.1 Labels: TRUE labels only

- `Data/train.csv` (46,713 rows) and `Data/val.csv` (5,193 rows) are
  **pseudo/generated labels — DO NOT use their labels for supervision, model
  selection, or evaluation.**
- The **only** trusted true-labeled pool is `Data/test.csv` (5,767 rows). Every
  labeled thing here — training, validation, final test — is carved out of
  `Data/test.csv`. See Section 3.

### 1.2 The quantum path takes RAW PIXELS only

- The Patch-QCNN input pipeline is: `raw image → fixed normalization → fixed
  patch extraction → quantum encoding`. Nothing learned or classical-parametric
  may sit between the raw pixels and the quantum circuit.
- **Do NOT feed E0/E1/E2/E3 embeddings, PCA, or any learned/fitted classical
  feature extractor into the quantum path.** That would recreate the exact "E3 is
  just a head behind a classical encoder" problem this experiment exists to
  escape. Fixed, data-independent preprocessing (per-image normalization, fixed
  patch grid, fixed Morton/Z-order pixel reindexing) is allowed — it learns
  nothing.
- **E0 is allowed ONLY as an external reference benchmark** (Section 8, baseline
  B-E0), never inside the QCNN pipeline. When you report E0, disclose that E0 was
  trained on pseudo-labels but is *graded* on true test labels.

### 1.3 No leakage

- Fit every transform (normalization statistics, PCA used *only inside classical
  baselines*, any scaler) **only on the trusted-train split**; apply to val/test,
  never re-fit.
- Corruption is applied **only to val/test images at evaluation time**
  (Section 9). Training images stay clean unless a baseline is explicitly a
  *corruption-trained* variant (B-CT), which trains on its own corruption-
  augmented copy of the SAME train split — never on val/test.

### 1.4 Do not touch report deliverables

- Only create Markdown / Python / result CSV-JSON files inside
  `full_qcnn_pilot/`. Do not edit any `.docx` or `.pdf` anywhere in the repo.

### 1.5 Fairness is mandatory, not optional

- Every model (quantum and classical) trains and evaluates on **identical**
  splits, images, corruptions (if the corruption axis is on), severities,
  augmentation policy, and metrics.
- The **quadratic null** (B-QUAD, Section 8) and the **parameter-matched patch
  classical twin** (B-PCNN, Section 8) are REQUIRED regardless of task or whether
  corruption runs. A QCNN result not reported alongside B-QUAD and B-PCNN is not a
  result. (B-CT, the corruption-trained baseline, is required only when the
  corruption axis is on.)

---

## 2. Environment

- Python venv: `WORKSPACE/.venv-e1/Scripts/python.exe` (Python 3.10,
  torch 2.x + CUDA, scikit-learn, numpy, pandas, PIL/Pillow available).
- Quantum framework: **PennyLane with the `default.qubit` simulator and the
  PyTorch interface** (`qml.qnn.TorchLayer`), `diff_method="backprop"`. Install
  into the venv if missing:
  ```powershell
  & "WORKSPACE/.venv-e1/Scripts/python.exe" -m pip install pennylane pennylane-lightning
  ```
  **IMPORTANT (measured 2026-07-11):** use `default.qubit` + `backprop`, NOT
  `lightning.qubit` + `adjoint`, for the patch layer. `default.qubit` **broadcasts**
  the 196 per-patch circuits as ONE vectorized tensor op; `lightning.qubit` runs
  them in a Python loop (one circuit at a time). Measured on this workload:
  patch-layer forward 60 ms (broadcast) vs 2,770 ms (per-sample loop) — **46×**.
  Full model fwd+bwd ≈ 147 ms/image → **~2.7 min/epoch** on the 1,077-image
  crop3 task. Feed batched inputs shaped `[batch*196, 1024]` so broadcasting
  engages. `pennylane-lightning` may still be installed but is not the default path.
- Reuse the existing corruption code — do NOT rewrite the corruption math:
  - Image corruption: `corruption_study/gen/corrupt_images.py`
    (Gaussian noise, motion blur, defocus blur, contrast, brightness, JPEG).
- Windows + PowerShell. Run everything through the venv interpreter above.

---

## 3. Data and splits

### 3.1 Source and label parsing

- Trusted pool: `Data/test.csv`. **Raw `Data/test.csv` has only two columns:
  `image` and `label`** (e.g. `10027,3_b7_1`).
- Derive `crop, disease, stage = label.split("_")` — exactly three tokens.
  Assert `len(label.split("_")) == 3` and fail loudly otherwise. `crop` and
  `stage` are integers; `disease` is a code (`00` healthy, else e.g. `a9`, `b7`).
- Images live under `Data/test/<image_id>/<image_id>.jpg` (VERIFIED 2026-07-11:
  each test id has its own folder `Data/test/<id>/` containing `<id>.jpg` plus IoT
  files `<id>.csv`, `<id>.json`, `<id>_final.csv`; all 5,767 ids resolve). Use
  this exact path template; `scripts/build_splits.py` already resolves and
  optionally verifies it. Confirm every selected image file exists before
  training; fail loudly on any missing file.

### 3.2 The pilot chain (the 3-class head)

- Use ONLY the `3_a9` ordinal chain:
  - `3_a9_1` (154 images) → stage 1
  - `3_a9_2` (111 images) → stage 2
  - `3_a9_3` (42 images)  → stage 3
  - Total: **307 images**. This tiny size is what makes full-resolution circuit
    simulation tractable — do not silently expand it.
- Labels are the integer stage `{1,2,3}` remapped to `{0,1,2}`. This is an
  **ordinal** target; report both plain Macro-F1 and stage-boundary metrics
  (Section 10).
- Class imbalance is real (154/111/42). Handle it in *training* (class-weighted
  loss or balanced sampling) but **never** with `class_weight`-style probability
  reweighting on the evaluation path, because that corrupts calibration (ECE).
  Keep an unreweighted probability source for all calibration metrics.

### 3.3 Splits

- Stratified by stage into **train / val / test = 60 / 20 / 20** within each
  stage. Val is for early stopping / hyperparameter selection ONLY; test is
  touched once for final numbers.
- **3 split seeds** `{0,1,2}`. Because N is small, report mean ± std across seeds
  for every metric; a result that appears in only one seed is not a result.
- Write splits as JSON (`splits/split_seed{0,1,2}.json`, listing image ids per
  train/val/test) so every model consumes identical splits.

### 3.4 Why a reduced head is legitimate

The architectural novelty lives in the **encoder** (how 224×224 pixels enter and
are processed by quantum circuits), which is independent of the number of output
classes. A 3-way head keeps the build tractable and forces the model to learn
*severity* rather than crop/disease identity. Expanding to more classes is
Phase 4 and only if a signal survives (Section 11). State this explicitly in the
writeup so the novelty claim is not misread as "only 3 classes."

---

## 4. Phase 0 — resolution sanity (do this BEFORE building the QCNN)

Confirm disease-stage information actually needs resolution, so the pilot is not
secretly a downscaled toy. Train a simple classical model (a small from-scratch
CNN, or logistic regression on flattened pixels) on the `3_a9` chain at:

```text
28×28   56×56   112×112   224×224
```

Report Macro-F1 vs resolution. Expected: performance rises with resolution
(stage detail lives in fine texture). If performance is already saturated at
28×28, note it — it would mean the task does not need full resolution and the
full-res framing is weaker. Record the curve in `results/phase0_resolution.csv`.

---

## 5. Quantum encoding spec

### 5.1 Patch grid

- Fixed division of the 224×224×3 image into **16×16 patches** → a **14×14 grid
  of 196 patches**, each `16×16×3 = 768` values. No overlap, no learned stride.
- Per-image normalization only: standardize each image to zero-mean/unit-var
  using statistics computed *per image* (data-independent, learns nothing), then
  min-max into `[0, π]` for angle steps as needed.

### 5.2 Per-patch amplitude encoding (primary scheme)

- Amplitude-encode each patch's 768 values into a **10-qubit** register
  (`2^10 = 1024`, zero-pad the last 256 amplitudes). Normalize the 768-vector to
  unit L2 before encoding (amplitude encoding requires it). **Note in the writeup
  that this normalization discards per-patch global magnitude** — this is a known
  limitation; the pooling levels partially recover cross-patch magnitude via the
  measured features.
- **Preserve 2D locality** by ordering the 768 amplitudes with a **Morton /
  Z-order curve** over the 16×16×3 patch so spatially adjacent pixels map to
  nearby basis states. Low-order qubits then correspond to fine detail, high-order
  qubits to coarse structure — design the convolution gates accordingly (5.4).
- Use `qml.AmplitudeEmbedding(..., normalize=True, pad_with=0.0)`.

### 5.3 Fallback scheme (if 10-qubit-per-patch is too slow)

- Encode each channel's `16×16 = 256` values into an **8-qubit** register
  (`2^8 = 256`, exact) and process the 3 channels with a shared circuit, then
  combine the 3 channel outputs at the first pooling level. Keeps RGB but reduces
  per-register width. Only switch to this if the primary scheme's per-epoch time
  exceeds the budget in Section 7.

### 5.4 Data re-uploading (to exceed the quadratic ceiling — honestly)

- A single amplitude-encoded circuit's measured feature is a **quadratic form in
  the patch amplitudes** (`f = xᵀAx`). To access a richer-than-quadratic Fourier
  spectrum, allow **L = 2–3 data re-uploading layers**: re-encode a fixed
  transform of the patch (e.g. re-embed with `qml.AngleEmbedding` of pooled patch
  summaries) between trainable blocks. Document L as a reported hyperparameter.
- Be explicit in the writeup: with L=1 the model is expected to be
  quadratic-equivalent; L>1 is the mechanism by which it can, in principle,
  exceed the quadratic null (B-QUAD). B-QUAD is the exact control that measures
  whether the extra layers buy anything.

### 5.5 Option A — magnitude-preserving encoding ablation (`amplitude10_mag`)

**Status: SPECIFIED 2026-07-11 after the Phase-A result. This is an encoding
ablation, not post-result circuit tuning: it changes only WHAT fixed,
data-independent information enters the circuit, not the variational
architecture, parameter count, training protocol, or selection rules.**

Motivation (from the completed Phase-A analysis): the L2 normalization in 5.2
deletes each patch's overall magnitude, while B-QUAD (which won Phase A at
0.7264 Macro-F1) classifies on absolute post-normalization channel means. This
ablation tests the hypothesis *"the magnitude discard is a material part of the
QCNN–B-QUAD gap"* by re-injecting exactly the discarded scalars — nothing more.

Design:

- Per patch, compute **4 fixed scalars** from the same normalized pixels the
  amplitude vector is built from (all data-independent, no learning):
  1. mean of the R channel (already in `[0, π]`),
  2. mean of the G channel,
  3. mean of the B channel,
  4. the patch's L2 norm before unit-normalization, rescaled to `[0, π]` by the
     fixed constant `π·√768` (the maximum possible norm given pixels in `[0, π]`).
- Encoding tensor per image: `[196, 1024 + 4]` — the existing Morton-ordered,
  zero-padded, L2-normalized amplitude vector concatenated with the 4 scalars.
- Circuit change: after `qml.AmplitudeEmbedding` of `inputs[..., :1024]`, apply
  `qml.RY(inputs[..., 1024 + j])` on the 4 readout wires (wires 6–9), j = 0..3,
  BEFORE the trainable layers. Re-uploading rounds (5.4) are unchanged.
- **Zero new trainable parameters** — the model stays at 133, so the existing
  B-PCNN parameter match and B-QUAD null remain valid without retraining them.

Fairness rules for this arm:

- B-QUAD and B-PCNN results are reused as-is (B-QUAD already consumes channel
  means; B-PCNN already consumes raw un-normalized-magnitude patches — neither
  baseline is disadvantaged by this change).
- Same splits (seeds 0/1/2), same training protocol (Section 7), same
  val-only selection, test touched once per seed.
- Report as a separate row `QCNN-mag` alongside the existing `QCNN` row; the
  original Phase-A QCNN result is NOT superseded — the pair IS the ablation.
- Honest expectation, stated in advance: this should close PART of the
  QCNN–B-QUAD gap. It cannot close all of it (function class stays ~quadratic at
  L=1; the 8-feature readout bottleneck is unchanged). If it closes little or
  none of the gap, that is a valid finding: the gap is attributable to function
  class/capacity, not the encoding.

---

## 6. Quantum circuit spec

### 6.1 Shared patch-convolution circuit (the "filter")

- One trainable circuit `U_conv(θ_conv)` applied to **every** patch register with
  **shared** parameters θ_conv (weight-sharing = the "convolution").
- Structure per layer: a ring of parameterized single-qubit rotations
  (`RY`, `RZ`) followed by nearest-neighbour entangling gates (`CNOT` or
  `CZ`) on the qubit ring, respecting the Morton ordering so low/high-order
  qubits mix at appropriate scales. Depth `d_conv ∈ {2,3,4}` (reported).
- **Readout per patch:** measure `⟨Z⟩` on a fixed subset of **k = 4** designated
  "pooling qubits" (the coarse high-order qubits). Output: a `14×14×4` feature
  map. This measurement between levels is deliberate and is the trainable/simulable
  ("Branch A") design; the coherent all-patches-entangled design is out of scope
  for the pilot (Section 12).

### 6.2 Hierarchical quantum pooling

- **Level 1:** group the `14×14×4` map into non-overlapping `2×2` spatial blocks
  → `7×7` cells, each holding `2×2×4 = 16` measured features. Angle-encode each
  cell's 16 features into a **4-qubit** register (`2^4 = 16`) and apply a shared
  trainable pooling circuit `U_pool1(θ_pool1)`; measure `⟨Z⟩` on all 4 qubits →
  `7×7×4` map.
- **Level 2:** repeat with `7×7 → ` flatten the 49 cells' `4` features
  (`49×4 = 196` values) → angle/amplitude-encode into an **8-qubit** register and
  apply a shared trainable circuit `U_pool2(θ_pool2)`; measure `⟨Z⟩` on all 8
  qubits → a **global 8-D quantum feature vector**.
- All pooling circuits are trainable and shared within their level.

### 6.3 Readout head

- A **single linear layer** (softmax over 3 classes) on the global 8-D feature
  vector. Linear only — a linear readout on measured observables is standard and
  does not constitute learned classical feature extraction. No hidden layers, no
  nonlinearity before it, so the "representation learning is quantum" claim holds.

### 6.4 Parameter budget

- Keep total trainable parameters (θ_conv + θ_pool1 + θ_pool2 + linear head) in
  the **low hundreds**. Record the exact count — it defines the parameter-matched
  classical twin B-PCNN (Section 8).

---

## 7. Training protocol and simulation budget

- Framework: PennyLane `lightning.qubit`, PyTorch interface, `diff_method=
  "adjoint"`. CPU is fine for the pilot; use GPU (`lightning.gpu`) only if
  installed and faster.
- Loss: class-weighted cross-entropy (weights from train-split stage counts).
  Keep a second, **unweighted** forward pass available for calibration metrics.
- Optimizer: Adam, lr `1e-2` → cosine/step decay; batch size 8–16 images;
  early-stop on val Macro-F1 with patience ~15 epochs; max ~100 epochs.
- **Budget check (do this first, on seed 0, 1 epoch):** with `default.qubit`
  broadcast (Section 2) the full model is **measured at ~2.7 min/epoch** on the
  1,077-image crop3 task (fwd+bwd ≈ 147 ms/image), well under budget. The earlier
  ~48 min/epoch was the `lightning.qubit` per-sample loop and is resolved by the
  simulator switch — do NOT downscale the task or architecture to fix speed; fix
  it with `default.qubit` broadcast. If a smoke still exceeds ~20 min, only then
  consider the 5.3 fallback or reducing `d_conv`. Log measured per-epoch
  wall-clock in `RESULTS_LOG.md`.
- Save the best-val checkpoint per seed. Never select on test.

---

## 8. Classical baselines (ALL required, identical splits/training)

| id | model | purpose |
| --- | --- | --- |
| **B-E0** | E0 (224) penultimate features → fresh linear head trained on `3_a9` true labels | operational reference ceiling; disclose pseudo-label provenance; NOT in the quantum path |
| **B-PCNN** | classical patch-CNN twin: identical 16×16 patch grid, a shared small conv "filter" producing 4 features/patch, identical 2-level hierarchical pooling, **parameter-matched to the QCNN** (Section 6.4) | the fair architectural twin — the single most important comparison |
| **B-QUAD** | logistic regression / linear SVM on **explicit degree-2 features** of pooled patch means (or a Nyström/random-feature quadratic map) | the null the amplitude-encoded QCNN must beat to claim it learns beyond quadratic |
| **B-RBF** | RBF-kernel SVM on flattened downsampled pixels or patch summaries | standard strong non-linear classical baseline |
| **B-POLY2** | degree-2 polynomial-kernel SVM | kernel-space twin of the quadratic mechanism |
| **B-CNN** | small from-scratch CNN at 224×224 | conventional deep baseline |
| **B-CT** | corruption-trained version of the strongest of {B-PCNN, B-CNN} (trained on corruption-augmented copy of the SAME train split) | tests whether any QCNN corruption edge survives a classically corruption-hardened model |

Every baseline: same 3 seeds, same splits, same corruption eval, same metrics.
Report all in one table alongside the QCNN.

---

## 9. Corruption protocol

- Reuse `corruption_study/gen/corrupt_images.py`. Apply corruption to **val/test
  images only** (except B-CT, which augments its own train split).
- Corruptions: `gaussian_noise`, `defocus_blur`, `motion_blur`, `contrast`,
  `jpeg`. Severities `{2, 3, 5}`.
- For each (model, corruption, severity, seed) record the full metric set. Also
  record clean (severity 0).
- Corruption seeds fixed and shared across all models so everyone sees identical
  corrupted images.

---

## 10. Metrics

Per (model, condition, seed), then aggregated mean ± std over seeds:

- Macro-F1 (primary), accuracy.
- **Clean-to-corrupted drop** = clean Macro-F1 − corrupted Macro-F1 (per
  corruption and mean over corruptions). Lower is better; this is the core
  robustness quantity.
- Worst-corruption Macro-F1.
- **Stage-boundary recall** (recall on adjacent-stage confusions, esp. the known
  `3_a9_2 ↔ 3_a9_1` failure) and per-class recall.
- Calibration: **ECE (15-bin)**, Brier score, NLL — all from the *unweighted*
  probability source.

Write `results/all_runs.csv` (one row per model×condition×severity×seed) and
`results/summary.csv` (aggregated). Produce a `results/QCNN_REPORT.md` with the
headline table: QCNN vs B-PCNN vs B-QUAD vs B-E0 across clean (+ corruptions if on).

### 10.1 Required analysis studies (these ARE the positive contribution)

Because the project claim is the architecture, these two studies are first-class
deliverables — run them even if the QCNN never beats a classical baseline.

**A. Scalability / ablation study** → `results/scalability.csv`. Vary one factor
at a time and record clean Macro-F1, trainable-param count, and per-epoch
wall-clock:
- patch size: `8×8` vs `16×16` (grid 28×28 vs 14×14);
- qubits per patch (encoding width);
- conv depth `d_conv ∈ {2,3,4}`;
- data re-uploading rounds `L ∈ {1,2,3}`;
- quantum pooling vs plain classical averaging of patch features;
- shared vs per-patch parameters (expect shared to generalize better at small N).
Report the accuracy–parameter–simulation-cost trade-off curve.

**B. Function-class study** → `results/function_class.md`. Determine what the
quantum patch layer actually computes by comparing the QCNN against B-QUAD
(degree-2) and B-POLY2 on identical inputs, and by checking whether `L=1` matches
a quadratic surrogate while `L>1` diverges from it. Conclude which of {linear,
quadratic, polynomial-kernel, richer-via-reuploading} best describes the learned
map. This is academically interesting independent of accuracy, and it is the
honest answer to "is the quantum structure doing anything?"

---

## 11. Go / no-go — framed around the ARCHITECTURE claim

The project contribution is the architecture + its honest characterization, so the
gate decides **whether to EXPAND** (Phase 4: more chains/classes, coherent
pooling), NOT whether the project lives. The pilot is publishable/defensible
either way as "we built and characterized the first full-resolution Patch-QCNN
for crop disease staging."

**Expand (Phase 4)** only if, across all 3 seeds:

1. Disease-stage info is available at full resolution (Phase 0 confirms).
2. QCNN clean Macro-F1 is within ~2–3 points of B-PCNN (competitive, not
   necessarily winning).
3. QCNN clean-to-corrupted drop < B-PCNN drop on **≥ 2** corruptions.
4. The corruption edge repeats across all 3 seeds.
5. The edge survives **B-CT** (corruption-trained classical).
6. The edge survives **B-QUAD** and **B-POLY2** (i.e. the QCNN is doing something
   beyond the quadratic function class).
7. Any gain shows in stage-boundary / per-class recall, not only majority-class
   accuracy.

**Do not expand (close the win-hunt, keep the architecture result)** if:

- QCNN sits far below B-PCNN/B-E0 on clean, OR
- an edge appears only vs weak baselines, only in one seed, or only in one
  corruption, OR
- the edge vanishes once B-QUAD/B-POLY2/B-CT are added, OR
- you find yourself repeatedly deepening circuits to manufacture a win.

In the "do not expand" case, the writeup reports the architecture, the Phase 0
resolution curve, the full fair-baseline table, and the honest finding — that is
a complete positive contribution (novel architecture + rigorous characterization),
**not** a failure.

---

## 12. Scope boundary — what is IN and OUT for the pilot

**IN:** full 224×224 images, fixed 16×16 patch grid, per-patch amplitude
encoding with Morton ordering, shared conv circuit, measured-between-levels
hierarchical pooling ("Branch A"), 3-class `3_a9` head, the full baseline set,
corruption eval, go/no-go.

**OUT (Phase 2+ stretch, only after a Phase-1 signal):**
- Coherent all-patches-entangled pooling with no intermediate measurement
  ("Branch B") — this is where higher-order expressivity *and* barren-plateau /
  simulability tension live; it is a separate research effort, not the pilot.
- IoT fusion (entangling an IoT register with the image register).
- More chains / all 25 classes.
- Quantum Fourier / wavelet pooling branches.

Do not silently pull Phase-2 items into the pilot.

---

## 13. Risks and honest expected outcome

- Amplitude state preparation and 196-patch simulation are the cost drivers;
  the tiny 307-image chain is what keeps it feasible. Guard the budget (Section 7).
- Normalization discards per-patch magnitude; barren-plateau risk exists but
  QCNN-style architectures have shown polynomially- (not exponentially-) vanishing
  gradients — monitor gradient norms and report them.
- Honest expectation: **beating B-E0 on clean images is unlikely; a narrow,
  possibly-real corruption or calibration edge is possible; a broad uniquely-
  quantum advantage is low probability.** The B-QUAD null is the honest arbiter of
  whether the quantum structure contributes beyond a quadratic classifier.

---

## 14. Deliverables / file layout

```text
full_qcnn_pilot/
  README.md               # how to run, in order
  RUN_COMMANDS.md         # exact PowerShell command lines
  RESULTS_LOG.md          # dated run notes, per-epoch wall-clock, interpretation
  CHANGELOG.md            # every change to this folder
  scripts/
    build_splits.py       # Section 3: carve 3_a9 chain, stratified, 3 seeds
    phase0_resolution.py  # Section 4: classical Macro-F1 vs resolution
    patch_encode.py       # Section 5: patch grid, Morton order, amplitude embed
    qcnn_model.py         # Section 6: conv circuit, pooling levels, linear head
    train_qcnn.py         # Section 7: training loop + budget smoke
    baselines.py          # Section 8: B-E0/B-PCNN/B-QUAD/B-RBF/B-POLY2/B-CNN/B-CT
    corrupt_eval.py       # Section 9: reuse corrupt_images.py, val/test only
    metrics.py            # Section 10
    run_experiment.py     # orchestrates everything, writes results CSVs + report
  splits/                 # split_seed{0,1,2}.json
  results/                # all_runs.csv, summary.csv, phase0_resolution.csv, QCNN_REPORT.md
```

Update `CHANGELOG.md` on every change and `RESULTS_LOG.md` on every run, exactly
as the `progression_graph_experiment/` folder does.

---

## 15. Naming

This system replaces E0 (learns directly from pixels), so it is not the E3 head.
Working name for the project:

```text
Q0: Full-Resolution Hierarchical Patch-QCNN
```

Alternative labels if a different convention is preferred: `Patch-QCNN`,
`E3-FQ` (full-quantum E3), `HQ-CropNet`. Pick one and use it consistently.

---

## 16. Rationale appendix (why this design, in brief)

- **Why patches, not one 18-qubit flat image:** a single arbitrary 18-qubit
  amplitude state stores the image but has no built-in 2D locality; patches +
  shared circuit + hierarchical pooling reintroduce convolutional structure.
- **Why measured-between-levels (Branch A):** it is trainable and classically
  simulable at pilot scale. Its honest cost is that each level is low-order in its
  inputs; data re-uploading (5.4) and multiple levels are the only ways it exceeds
  quadratic, and B-QUAD measures whether they do.
- **Why the 3-class chain:** novelty is in the encoder, not the head; a small
  ordinal head makes 224×224 simulation feasible and forces severity learning.
- **Why the win is not the point:** three prior fair tests (VQC head, quantum-walk,
  and the literature precedents where quantum vision ties/loses to fed classical
  baselines) make a broad advantage low-probability. The defensible, positive
  contribution is the architecture and its rigorous, fairly-baselined
  characterization under corruption.
```
