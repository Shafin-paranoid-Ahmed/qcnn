# Phase A3 Attention Report

Generated: 2026-07-16

All values are mean ± sample standard deviation across seeds 0/1/2. Test data were evaluated once per seed after validation-only checkpoint selection.

| Arm | Params | Macro-F1 | Accuracy | ECE | Brier | NLL | Macro recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QCNN-attn | 138 | 0.5068 ± 0.0542 | 0.7084 ± 0.0326 | 0.0825 ± 0.0029 | 0.4024 ± 0.0390 | 0.8060 ± 0.0688 | 0.5170 ± 0.0638 |
| B-PCNN-attn | 138 | 0.3793 ± 0.0910 | 0.6370 ± 0.0137 | 0.1726 ± 0.0977 | 0.5362 ± 0.0812 | 1.0468 ± 0.1497 | 0.4060 ± 0.0912 |
| B-ATTN | 24 | 0.3493 ± 0.0642 | 0.6527 ± 0.0307 | 0.2858 ± 0.0372 | 0.6100 ± 0.0463 | 1.2073 ± 0.0862 | 0.3560 ± 0.0586 |
| QCNN-mag | 133 | 0.5442 ± 0.0473 | 0.7038 ± 0.0227 | 0.1245 ± 0.0246 | 0.4129 ± 0.0518 | 0.8089 ± 0.1016 | 0.5814 ± 0.0454 |
| B-PCNN | 133 | 0.2922 ± 0.0078 | 0.6825 ± 0.0195 | 0.2490 ± 0.0445 | 0.5678 ± 0.0172 | 1.1304 ± 0.0184 | 0.3200 ± 0.0130 |
| B-QUAD | n.a. | 0.7264 ± 0.0198 | 0.8319 ± 0.0216 | 0.1116 ± 0.0174 | 0.2860 ± 0.0349 | 0.9653 ± 0.0444 | 0.7155 ± 0.0204 |

## Pre-registered verdict

QCNN-attn − QCNN-mag mean Macro-F1: -0.0374. The ±0.03 expectation **did not hold**.
QCNN-attn − B-PCNN-attn mean Macro-F1: +0.1274. QCNN-attn **stayed ahead**, against the pre-registered expectation.
B-ATTN − QCNN-attn mean Macro-F1: -0.1575; B-ATTN − B-PCNN-attn: -0.0300. B-ATTN **was below both gated hierarchical arms**. No numeric threshold for 'NEAR' was pre-registered, so none is added post hoc.
**B-ATTN 'NEAR' qualitative verdict: pending human review.** If B-ATTN is judged near QCNN-attn, record the pre-registered conclusion that attention, not the quantum hierarchy, carries the gain.
Level-2 trigger (QCNN-attn gain ≥ +0.02): **not met; remains parked**.

## Saliency figure index

| Arm | True class | Outcome | Rank | Image | Predicted | Confidence | Alpha CSV | Overlay PNG |
| --- | --- | --- | ---: | --- | --- | ---: | --- | --- |
| qcnn_attn | 3_00_0 | correct | 1 | 61909 | 3_00_0 | 0.9301 | `results\saliency\qcnn_attn_L1_mag_seed0_3_00_0_correct1_61909.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_00_0_correct1_61909.png` |
| qcnn_attn | 3_00_0 | correct | 2 | 63576 | 3_00_0 | 0.9207 | `results\saliency\qcnn_attn_L1_mag_seed0_3_00_0_correct2_63576.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_00_0_correct2_63576.png` |
| qcnn_attn | 3_00_0 | correct | 3 | 40521 | 3_00_0 | 0.9201 | `results\saliency\qcnn_attn_L1_mag_seed0_3_00_0_correct3_40521.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_00_0_correct3_40521.png` |
| qcnn_attn | 3_00_0 | misclassified | 1 | 19471 | 3_b3_1 | 0.5158 | `results\saliency\qcnn_attn_L1_mag_seed0_3_00_0_misclassified1_19471.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_00_0_misclassified1_19471.png` |
| qcnn_attn | 3_00_0 | misclassified | 2 | 47061 | 3_b8_1 | 0.4862 | `results\saliency\qcnn_attn_L1_mag_seed0_3_00_0_misclassified2_47061.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_00_0_misclassified2_47061.png` |
| qcnn_attn | 3_b3_1 | correct | 1 | 31913 | 3_b3_1 | 0.6138 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b3_1_correct1_31913.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b3_1_correct1_31913.png` |
| qcnn_attn | 3_b3_1 | correct | 2 | 26898 | 3_b3_1 | 0.5130 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b3_1_correct2_26898.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b3_1_correct2_26898.png` |
| qcnn_attn | 3_b3_1 | correct | 3 | 15071 | 3_b3_1 | 0.4950 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b3_1_correct3_15071.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b3_1_correct3_15071.png` |
| qcnn_attn | 3_b3_1 | misclassified | 1 | 39195 | 3_00_0 | 0.7825 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b3_1_misclassified1_39195.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b3_1_misclassified1_39195.png` |
| qcnn_attn | 3_b3_1 | misclassified | 2 | 17324 | 3_00_0 | 0.6804 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b3_1_misclassified2_17324.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b3_1_misclassified2_17324.png` |
| qcnn_attn | 3_b6_1 | misclassified | 1 | 62218 | 3_00_0 | 0.8960 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b6_1_misclassified1_62218.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b6_1_misclassified1_62218.png` |
| qcnn_attn | 3_b6_1 | misclassified | 2 | 67636 | 3_00_0 | 0.8749 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b6_1_misclassified2_67636.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b6_1_misclassified2_67636.png` |
| qcnn_attn | 3_b7_1 | correct | 1 | 13249 | 3_b7_1 | 0.8820 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b7_1_correct1_13249.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b7_1_correct1_13249.png` |
| qcnn_attn | 3_b7_1 | correct | 2 | 12961 | 3_b7_1 | 0.8804 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b7_1_correct2_12961.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b7_1_correct2_12961.png` |
| qcnn_attn | 3_b7_1 | correct | 3 | 34188 | 3_b7_1 | 0.8539 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b7_1_correct3_34188.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b7_1_correct3_34188.png` |
| qcnn_attn | 3_b7_1 | misclassified | 1 | 19906 | 3_00_0 | 0.8553 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b7_1_misclassified1_19906.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b7_1_misclassified1_19906.png` |
| qcnn_attn | 3_b7_1 | misclassified | 2 | 43726 | 3_b8_1 | 0.6118 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b7_1_misclassified2_43726.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b7_1_misclassified2_43726.png` |
| qcnn_attn | 3_b8_1 | correct | 1 | 36780 | 3_b8_1 | 0.6606 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b8_1_correct1_36780.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b8_1_correct1_36780.png` |
| qcnn_attn | 3_b8_1 | correct | 2 | 10251 | 3_b8_1 | 0.5717 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b8_1_correct2_10251.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b8_1_correct2_10251.png` |
| qcnn_attn | 3_b8_1 | correct | 3 | 35402 | 3_b8_1 | 0.5670 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b8_1_correct3_35402.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b8_1_correct3_35402.png` |
| qcnn_attn | 3_b8_1 | misclassified | 1 | 44839 | 3_00_0 | 0.6479 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b8_1_misclassified1_44839.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b8_1_misclassified1_44839.png` |
| qcnn_attn | 3_b8_1 | misclassified | 2 | 61137 | 3_00_0 | 0.6000 | `results\saliency\qcnn_attn_L1_mag_seed0_3_b8_1_misclassified2_61137.csv` | `results\figures\qcnn_attn_L1_mag_seed0_3_b8_1_misclassified2_61137.png` |
| bpcnn_attn | 3_00_0 | correct | 1 | 52249 | 3_00_0 | 0.9644 | `results\saliency\bpcnn_attn_seed0_3_00_0_correct1_52249.csv` | `results\figures\bpcnn_attn_seed0_3_00_0_correct1_52249.png` |
| bpcnn_attn | 3_00_0 | correct | 2 | 47185 | 3_00_0 | 0.9532 | `results\saliency\bpcnn_attn_seed0_3_00_0_correct2_47185.csv` | `results\figures\bpcnn_attn_seed0_3_00_0_correct2_47185.png` |
| bpcnn_attn | 3_00_0 | correct | 3 | 23358 | 3_00_0 | 0.9510 | `results\saliency\bpcnn_attn_seed0_3_00_0_correct3_23358.csv` | `results\figures\bpcnn_attn_seed0_3_00_0_correct3_23358.png` |
| bpcnn_attn | 3_00_0 | misclassified | 1 | 10525 | 3_b7_1 | 0.9043 | `results\saliency\bpcnn_attn_seed0_3_00_0_misclassified1_10525.csv` | `results\figures\bpcnn_attn_seed0_3_00_0_misclassified1_10525.png` |
| bpcnn_attn | 3_00_0 | misclassified | 2 | 42075 | 3_b7_1 | 0.7452 | `results\saliency\bpcnn_attn_seed0_3_00_0_misclassified2_42075.csv` | `results\figures\bpcnn_attn_seed0_3_00_0_misclassified2_42075.png` |
| bpcnn_attn | 3_b3_1 | correct | 1 | 43465 | 3_b3_1 | 0.4010 | `results\saliency\bpcnn_attn_seed0_3_b3_1_correct1_43465.csv` | `results\figures\bpcnn_attn_seed0_3_b3_1_correct1_43465.png` |
| bpcnn_attn | 3_b3_1 | correct | 2 | 58418 | 3_b3_1 | 0.3824 | `results\saliency\bpcnn_attn_seed0_3_b3_1_correct2_58418.csv` | `results\figures\bpcnn_attn_seed0_3_b3_1_correct2_58418.png` |
| bpcnn_attn | 3_b3_1 | correct | 3 | 29667 | 3_b3_1 | 0.3799 | `results\saliency\bpcnn_attn_seed0_3_b3_1_correct3_29667.csv` | `results\figures\bpcnn_attn_seed0_3_b3_1_correct3_29667.png` |
| bpcnn_attn | 3_b3_1 | misclassified | 1 | 39195 | 3_00_0 | 0.8400 | `results\saliency\bpcnn_attn_seed0_3_b3_1_misclassified1_39195.csv` | `results\figures\bpcnn_attn_seed0_3_b3_1_misclassified1_39195.png` |
| bpcnn_attn | 3_b3_1 | misclassified | 2 | 65073 | 3_00_0 | 0.7511 | `results\saliency\bpcnn_attn_seed0_3_b3_1_misclassified2_65073.csv` | `results\figures\bpcnn_attn_seed0_3_b3_1_misclassified2_65073.png` |
| bpcnn_attn | 3_b6_1 | correct | 1 | 64029 | 3_b6_1 | 0.5095 | `results\saliency\bpcnn_attn_seed0_3_b6_1_correct1_64029.csv` | `results\figures\bpcnn_attn_seed0_3_b6_1_correct1_64029.png` |
| bpcnn_attn | 3_b6_1 | correct | 2 | 29886 | 3_b6_1 | 0.4535 | `results\saliency\bpcnn_attn_seed0_3_b6_1_correct2_29886.csv` | `results\figures\bpcnn_attn_seed0_3_b6_1_correct2_29886.png` |
| bpcnn_attn | 3_b6_1 | correct | 3 | 65312 | 3_b6_1 | 0.4060 | `results\saliency\bpcnn_attn_seed0_3_b6_1_correct3_65312.csv` | `results\figures\bpcnn_attn_seed0_3_b6_1_correct3_65312.png` |
| bpcnn_attn | 3_b6_1 | misclassified | 1 | 56783 | 3_00_0 | 0.8014 | `results\saliency\bpcnn_attn_seed0_3_b6_1_misclassified1_56783.csv` | `results\figures\bpcnn_attn_seed0_3_b6_1_misclassified1_56783.png` |
| bpcnn_attn | 3_b6_1 | misclassified | 2 | 62218 | 3_00_0 | 0.7852 | `results\saliency\bpcnn_attn_seed0_3_b6_1_misclassified2_62218.csv` | `results\figures\bpcnn_attn_seed0_3_b6_1_misclassified2_62218.png` |
| bpcnn_attn | 3_b7_1 | correct | 1 | 20120 | 3_b7_1 | 0.9837 | `results\saliency\bpcnn_attn_seed0_3_b7_1_correct1_20120.csv` | `results\figures\bpcnn_attn_seed0_3_b7_1_correct1_20120.png` |
| bpcnn_attn | 3_b7_1 | correct | 2 | 12961 | 3_b7_1 | 0.9618 | `results\saliency\bpcnn_attn_seed0_3_b7_1_correct2_12961.csv` | `results\figures\bpcnn_attn_seed0_3_b7_1_correct2_12961.png` |
| bpcnn_attn | 3_b7_1 | correct | 3 | 18798 | 3_b7_1 | 0.9516 | `results\saliency\bpcnn_attn_seed0_3_b7_1_correct3_18798.csv` | `results\figures\bpcnn_attn_seed0_3_b7_1_correct3_18798.png` |
| bpcnn_attn | 3_b7_1 | misclassified | 1 | 12827 | 3_00_0 | 0.8082 | `results\saliency\bpcnn_attn_seed0_3_b7_1_misclassified1_12827.csv` | `results\figures\bpcnn_attn_seed0_3_b7_1_misclassified1_12827.png` |
| bpcnn_attn | 3_b7_1 | misclassified | 2 | 18248 | 3_b8_1 | 0.5067 | `results\saliency\bpcnn_attn_seed0_3_b7_1_misclassified2_18248.csv` | `results\figures\bpcnn_attn_seed0_3_b7_1_misclassified2_18248.png` |
| bpcnn_attn | 3_b8_1 | correct | 1 | 57996 | 3_b8_1 | 0.5409 | `results\saliency\bpcnn_attn_seed0_3_b8_1_correct1_57996.csv` | `results\figures\bpcnn_attn_seed0_3_b8_1_correct1_57996.png` |
| bpcnn_attn | 3_b8_1 | correct | 2 | 49040 | 3_b8_1 | 0.5203 | `results\saliency\bpcnn_attn_seed0_3_b8_1_correct2_49040.csv` | `results\figures\bpcnn_attn_seed0_3_b8_1_correct2_49040.png` |
| bpcnn_attn | 3_b8_1 | correct | 3 | 50940 | 3_b8_1 | 0.5067 | `results\saliency\bpcnn_attn_seed0_3_b8_1_correct3_50940.csv` | `results\figures\bpcnn_attn_seed0_3_b8_1_correct3_50940.png` |
| bpcnn_attn | 3_b8_1 | misclassified | 1 | 36780 | 3_b7_1 | 0.8531 | `results\saliency\bpcnn_attn_seed0_3_b8_1_misclassified1_36780.csv` | `results\figures\bpcnn_attn_seed0_3_b8_1_misclassified1_36780.png` |
| bpcnn_attn | 3_b8_1 | misclassified | 2 | 35402 | 3_b7_1 | 0.7232 | `results\saliency\bpcnn_attn_seed0_3_b8_1_misclassified2_35402.csv` | `results\figures\bpcnn_attn_seed0_3_b8_1_misclassified2_35402.png` |
| battn | 3_00_0 | correct | 1 | 18848 | 3_00_0 | 0.4903 | `results\saliency\battn_seed0_3_00_0_correct1_18848.csv` | `results\figures\battn_seed0_3_00_0_correct1_18848.png` |
| battn | 3_00_0 | correct | 2 | 52972 | 3_00_0 | 0.4593 | `results\saliency\battn_seed0_3_00_0_correct2_52972.csv` | `results\figures\battn_seed0_3_00_0_correct2_52972.png` |
| battn | 3_00_0 | correct | 3 | 66412 | 3_00_0 | 0.4493 | `results\saliency\battn_seed0_3_00_0_correct3_66412.csv` | `results\figures\battn_seed0_3_00_0_correct3_66412.png` |
| battn | 3_00_0 | misclassified | 1 | 65083 | 3_b6_1 | 0.3371 | `results\saliency\battn_seed0_3_00_0_misclassified1_65083.csv` | `results\figures\battn_seed0_3_00_0_misclassified1_65083.png` |
| battn | 3_00_0 | misclassified | 2 | 61245 | 3_b6_1 | 0.3170 | `results\saliency\battn_seed0_3_00_0_misclassified2_61245.csv` | `results\figures\battn_seed0_3_00_0_misclassified2_61245.png` |
| battn | 3_b3_1 | misclassified | 1 | 11358 | 3_00_0 | 0.3719 | `results\saliency\battn_seed0_3_b3_1_misclassified1_11358.csv` | `results\figures\battn_seed0_3_b3_1_misclassified1_11358.png` |
| battn | 3_b3_1 | misclassified | 2 | 21979 | 3_00_0 | 0.3649 | `results\saliency\battn_seed0_3_b3_1_misclassified2_21979.csv` | `results\figures\battn_seed0_3_b3_1_misclassified2_21979.png` |
| battn | 3_b6_1 | correct | 1 | 10764 | 3_b6_1 | 0.2619 | `results\saliency\battn_seed0_3_b6_1_correct1_10764.csv` | `results\figures\battn_seed0_3_b6_1_correct1_10764.png` |
| battn | 3_b6_1 | correct | 2 | 53117 | 3_b6_1 | 0.2530 | `results\saliency\battn_seed0_3_b6_1_correct2_53117.csv` | `results\figures\battn_seed0_3_b6_1_correct2_53117.png` |
| battn | 3_b6_1 | correct | 3 | 65312 | 3_b6_1 | 0.2499 | `results\saliency\battn_seed0_3_b6_1_correct3_65312.csv` | `results\figures\battn_seed0_3_b6_1_correct3_65312.png` |
| battn | 3_b6_1 | misclassified | 1 | 49109 | 3_00_0 | 0.4172 | `results\saliency\battn_seed0_3_b6_1_misclassified1_49109.csv` | `results\figures\battn_seed0_3_b6_1_misclassified1_49109.png` |
| battn | 3_b6_1 | misclassified | 2 | 50164 | 3_00_0 | 0.4148 | `results\saliency\battn_seed0_3_b6_1_misclassified2_50164.csv` | `results\figures\battn_seed0_3_b6_1_misclassified2_50164.png` |
| battn | 3_b7_1 | correct | 1 | 63006 | 3_b7_1 | 0.3197 | `results\saliency\battn_seed0_3_b7_1_correct1_63006.csv` | `results\figures\battn_seed0_3_b7_1_correct1_63006.png` |
| battn | 3_b7_1 | correct | 2 | 11107 | 3_b7_1 | 0.3059 | `results\saliency\battn_seed0_3_b7_1_correct2_11107.csv` | `results\figures\battn_seed0_3_b7_1_correct2_11107.png` |
| battn | 3_b7_1 | correct | 3 | 34188 | 3_b7_1 | 0.2969 | `results\saliency\battn_seed0_3_b7_1_correct3_34188.csv` | `results\figures\battn_seed0_3_b7_1_correct3_34188.png` |
| battn | 3_b7_1 | misclassified | 1 | 45268 | 3_00_0 | 0.3202 | `results\saliency\battn_seed0_3_b7_1_misclassified1_45268.csv` | `results\figures\battn_seed0_3_b7_1_misclassified1_45268.png` |
| battn | 3_b7_1 | misclassified | 2 | 19906 | 3_00_0 | 0.2983 | `results\saliency\battn_seed0_3_b7_1_misclassified2_19906.csv` | `results\figures\battn_seed0_3_b7_1_misclassified2_19906.png` |
| battn | 3_b8_1 | correct | 1 | 11179 | 3_b8_1 | 0.2489 | `results\saliency\battn_seed0_3_b8_1_correct1_11179.csv` | `results\figures\battn_seed0_3_b8_1_correct1_11179.png` |
| battn | 3_b8_1 | correct | 2 | 56752 | 3_b8_1 | 0.2220 | `results\saliency\battn_seed0_3_b8_1_correct2_56752.csv` | `results\figures\battn_seed0_3_b8_1_correct2_56752.png` |
| battn | 3_b8_1 | misclassified | 1 | 16767 | 3_00_0 | 0.3286 | `results\saliency\battn_seed0_3_b8_1_misclassified1_16767.csv` | `results\figures\battn_seed0_3_b8_1_misclassified1_16767.png` |
| battn | 3_b8_1 | misclassified | 2 | 60579 | 3_00_0 | 0.3184 | `results\saliency\battn_seed0_3_b8_1_misclassified2_60579.csv` | `results\figures\battn_seed0_3_b8_1_misclassified2_60579.png` |

## Saliency qualitative verdict

**Reviewed 2026-07-16 (qcnn_attn seed-0 overlays; figure-based, no localization labels).** The pre-registered expectation — attention concentrating on lesion/leaf regions rather than background for diseased classes — is **partially borne out, and reported honestly as mixed**:

- **Diseased classes (3_b7_1, 3_b8_1):** attention does fall on the leaf lamina and lesion-bearing regions of the central leaf; the b8 correct-1 overlay in particular peaks on discolored leaf tissue rather than background. This supports the expectation.
- **Healthy class (3_00_0):** attention drifts to soil/substrate and background rather than the leaf — expected, since "healthy" has no lesion to fixate on, but it means the saliency story is class-dependent, not a uniform "always looks at the leaf."
- **Overall the maps are diffuse** (consistent with the near-uniform 5-parameter gate and the small accuracy effect): the model attends broadly, with leaf-concentrated peaks for diseased classes rather than sharp lesion localization.

Honest framing for the project: the saliency artifact is a defensible interpretability output that shows the gate learns leaf-vs-background structure for diseased classes, but it is not a precise lesion detector and no such claim is made. Saliency concentration is a qualitative, figure-based observation only; no lesion localization labels were used.
