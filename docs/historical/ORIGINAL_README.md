> Historical thesis record. Original machine paths and commands are retained as provenance. Use the standalone [README](../../README.md) and runbook for current setup.

# Full-QCNN Pilot

Last updated: 2026-09-07.

Full-resolution hierarchical Patch-QCNN for crop-image classification.
The completed work supports architectural characterization, not quantum
advantage. Phase A, depth/magnitude ablations, and attention/saliency are
complete; Phase B corruption evaluation remains a stub and has not run.

## Completed results

Three-seed test Macro-F1, mean ± sample standard deviation:

| Model | Macro-F1 |
| --- | ---: |
| Base QCNN | 0.4481 ± 0.0637 |
| QCNN with magnitude preservation | 0.5442 ± 0.0473 |
| QCNN depth 2 + magnitude | 0.5464 ± 0.0345 |
| QCNN attention | 0.5068 ± 0.0542 |
| Corrected B-PCNN | 0.2922 ± 0.0078 |
| B-QUAD | 0.7264 ± 0.0198 |

Magnitude restoration helped; depth adds little on top of it. Attention
did not improve on QCNN-mag, and the Level-2 attention trigger was not met.
All QCNN variants remain below the stronger quadratic null.

Sources: [clean/magnitude report](../../results/QCNN_REPORT.md),
[function-class study](../../results/function_class.md),
[attention report](../../results/ATTENTION_REPORT.md), and
[results log](../../RESULTS_LOG.md).

## Implementation and data

The model, training, baseline, metric, orchestration, and attention code
under `scripts/` are implemented. The primary patch simulator is
`default.qubit` with `backprop` and batched broadcasting; the original
`lightning.qubit`/adjoint per-patch recommendation is superseded.

The five-class crop-3 splits come only from trusted `Data/test.csv`;
they are separate from legacy E0–E3 and SSL-HDC splits. Raw pixels enter the
quantum path, without a pretrained encoder or PCA. Report B-PCNN and B-QUAD
alongside QCNN results.

Use the [handoff](../../HANDOFF.md), [specification](../protocols/full_qcnn.md),
and [runbook](ORIGINAL_RUN_COMMANDS.md) for reproduction. Completed runs and preserved
split/checkpoint artifacts are not a fresh quick-start task. Use the explicit
project interpreter and pinned environment; no package installation is needed
merely to review results.
