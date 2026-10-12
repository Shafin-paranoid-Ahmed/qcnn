# Function-Class Study

B-QUAD is the explicit degree-2 null on identical fixed pooled raw-pixel
summaries: mean test Macro-F1 **0.7264** (seeds 0.7418 / 0.7334 / 0.7042).
The question this study answers: does adding re-upload depth (A2.1) or
restoring the magnitude the amplitude encoding discards (A2.2), or both
(A2.3), move the QCNN beyond that quadratic null?

## Mean ± std test Macro-F1 (3 seeds), all vs B-QUAD 0.7264

| Config | Encoding | L | Params | Test Macro-F1 | Δ vs L=1 |
|---|---|---|---|---|---|
| Phase-A baseline | amplitude10 | 1 | 133 | 0.4481 ± 0.0637 | — |
| A2.1 | amplitude10 | 2 | 173 | 0.4992 ± 0.0464 | +0.0511 |
| A2.1 | amplitude10 | 3 | 213 | 0.4783 ± 0.0797 | +0.0302 |
| A2.2 | amplitude10_mag | 1 | 133 | 0.5442 ± 0.0473 | +0.0961 |
| A2.3 | amplitude10_mag | 2 | 173 | 0.5464 ± 0.0345 | +0.0983 |
| **B-QUAD null** | (raw pooled) | — | — | **0.7264** | — |

## Verdict

**Neither re-uploading depth, magnitude restoration, nor their combination
moves the QCNN beyond the quadratic null.** The best configuration
(A2.3, L=2 + magnitude) reaches 0.5464 — still **0.180 below B-QUAD** — and
every configuration trails it by a wide, seed-robust margin.

Reading the two effects, consistent with the pre-registered expectations:

- **Depth (A2.1) gives a modest gain** and does not scale: L=2 adds +0.051
  over L=1, but L=3 gives back part of it (+0.030, within seed noise), and
  L=3's larger parameter count (213) buys nothing. Re-uploading only the
  first 10 Morton-ordered normalized values per patch (the pre-registered
  implementation limitation) is a thin signal, as expected.
- **Magnitude restoration (A2.2) is the larger single effect** (+0.096),
  because it returns exactly the absolute color/brightness signal B-QUAD
  classifies on and the amplitude encoding L2-normalizes away — matching
  the A2.2 expectation that it would beat A2.1.
- **The two effects do NOT stack (A2.3).** L=2 + magnitude (0.5464) is
  statistically indistinguishable from magnitude alone at L=1 (0.5442);
  combining depth and magnitude adds nothing over magnitude by itself. The
  bottleneck is not representational depth — it is the 8-feature readout
  and the residual quadratic ceiling, both of which persist under all
  ablations.

**Function-class conclusion:** on these crop codes the QCNN, even with
re-uploading and magnitude restoration, stays inside the degree-2 function
class that B-QUAD already occupies more effectively. The quantum model's
disadvantage is not fixed by depth or by recovering the discarded
magnitude; it is structural to the amplitude-encoding + shallow-readout
design. This is a clean negative function-class result, reported as the
finding.
