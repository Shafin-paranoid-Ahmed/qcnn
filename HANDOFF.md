> Historical thesis record. Original machine paths and commands are retained as provenance. Use the standalone [README](README.md) and runbook for current setup.

# Full-QCNN Pilot Handoff

Last updated: 2026-09-07. This replaces the original scaffold/build checklist.

## Read first

1. [Specification](docs/protocols/full_qcnn.md), including dated amendments.
2. [Current README](docs/historical/ORIGINAL_README.md), [results log](RESULTS_LOG.md), and [runbook](docs/historical/ORIGINAL_RUN_COMMANDS.md).
3. [Function-class results](results/function_class.md) and [attention results](results/ATTENTION_REPORT.md).

## Completed

Phase A, the corrected B-PCNN comparator, depth 2/3, magnitude preservation,
the conditional depth-2-plus-magnitude run, and all three attention arms
with seed-0 saliency have completed. Their three-seed artifacts are present.

Best combined QCNN test Macro-F1: 0.5464 ± 0.0345; QCNN-attn:
0.5068 ± 0.0542; B-QUAD: 0.7264 ± 0.0198. These support architecture and
function-class analysis, not quantum advantage. The attention Level-2
trigger was not met.

## Code status

| Area | State |
| --- | --- |
| Splits and resolution study | Implemented; existing split manifests/results must be preserved. |
| Patch encoding and QCNN | Implemented, including depth and magnitude variants. |
| Training, metrics, baselines, orchestration | Implemented and exercised in completed studies. |
| Attention, saliency, finalization | Implemented; Phase A3 complete. |
| `scripts/corrupt_eval.py` | Phase B stub raising `NotImplementedError`; no Phase-B result. |

The patch fast path uses PennyLane `default.qubit`, `backprop`, and
batched broadcasting. The original per-sample `lightning.qubit`/adjoint
instruction is superseded. Use the explicit interpreter under the actual
checkout and the pinned stack; do not install unpinned packages from an old
scaffold instruction.

## Continuing constraints

Trusted labels come only from the existing `Data/test.csv`-derived
crop-3 split manifests (five classes; seeds 0/1/2). Raw pixels enter the
quantum circuit; no learned encoder or PCA precedes it. Preserve B-PCNN and
B-QUAD controls, validation-only checkpoint selection, and mean/sample-std
reporting. Do not tune on test outcomes.

Phase B and other parked extensions are not automatic next runs. Follow
the user's selected task and retain original preregistered expectations as
dated history. Record any new work in CHANGELOG, RESULTS_LOG, and central
context; keep command ownership and artifact protections in the runbook.
