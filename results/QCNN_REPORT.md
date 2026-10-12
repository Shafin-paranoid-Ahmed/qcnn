# Q0 Clean Phase-A and Magnitude-Ablation Report

True-label `crop3_disease` splits from `Data/test.csv` only.

Documentation refreshed 2026-09-07: this table covers the base/magnitude
comparison. Later completed depth/combined results are in
[function_class.md](function_class.md); attention results are in
[ATTENTION_REPORT.md](ATTENTION_REPORT.md). The duplicated per-class columns
in the earlier Markdown rendering have been collapsed; metric values are unchanged.

| Model | Macro-F1 | Accuracy | ECE | Brier | NLL | Macro recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QCNN | 0.4481 +/- 0.0637 | 0.6416 +/- 0.0580 | 0.0886 +/- 0.0203 | 0.4745 +/- 0.0509 | 0.9429 +/- 0.0862 | 0.4457 +/- 0.0573 |
| QCNN-mag | 0.5442 +/- 0.0473 | 0.7038 +/- 0.0227 | 0.1245 +/- 0.0246 | 0.4129 +/- 0.0518 | 0.8089 +/- 0.1016 | 0.5814 +/- 0.0454 |
| B-PCNN | 0.2922 +/- 0.0078 | 0.6825 +/- 0.0195 | 0.2490 +/- 0.0445 | 0.5678 +/- 0.0172 | 1.1304 +/- 0.0184 | 0.3200 +/- 0.0130 |
| B-QUAD | 0.7264 +/- 0.0198 | 0.8319 +/- 0.0216 | 0.1116 +/- 0.0174 | 0.2860 +/- 0.0349 | 0.9653 +/- 0.0444 | 0.7155 +/- 0.0204 |

## Per-class recall

| Model | 3_00_0 | 3_b3_1 | 3_b6_1 | 3_b7_1 | 3_b8_1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QCNN | 0.7881 +/- 0.0624 | 0.2929 +/- 0.1262 | 0.2989 +/- 0.1108 | 0.5484 +/- 0.0645 | 0.3000 +/- 0.0667 |
| QCNN-mag | 0.7952 +/- 0.0318 | 0.3737 +/- 0.0463 | 0.2989 +/- 0.2296 | 0.8280 +/- 0.0493 | 0.6111 +/- 0.1072 |
| B-PCNN | 0.9534 +/- 0.0278 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6022 +/- 0.0672 | 0.0444 +/- 0.0192 |
| B-QUAD | 0.9195 +/- 0.0320 | 0.5354 +/- 0.0631 | 0.5172 +/- 0.0345 | 0.8387 +/- 0.0559 | 0.7667 +/- 0.0882 |
