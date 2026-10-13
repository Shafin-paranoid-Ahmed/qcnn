"""Corruption evaluation (spec Section 9). STUB — PHASE B ONLY, implement LAST.

Do NOT implement until the base QCNN (Phase A) has cleared its go/no-go.

- Reuse corruption_study/gen/corrupt_images.py — do NOT rewrite the corruption math.
- Apply corruption to VAL/TEST images only (training stays clean, except B-CT which
  augments its OWN train split — never val/test).
- Corruptions: gaussian_noise, defocus_blur, motion_blur, contrast, jpeg.
  Severities {2,3,5}. Fixed shared corruption seeds so every model sees identical
  corrupted images.
- A null / fall-short corruption result is EXPECTED and ACCEPTABLE (spec 0.2).
  Corruption is NOT the project headline.
"""
from __future__ import annotations


def main():
    raise NotImplementedError("Phase B — implement after Phase A go/no-go (spec Section 9)")


if __name__ == "__main__":
    main()
