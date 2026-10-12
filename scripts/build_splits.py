"""Build trusted-label train/val/test splits for the Full-QCNN pilot.

TRUE LABELS ONLY: every id comes from Data/test.csv (the only trusted-labeled
pool). Labels parse as `crop_disease_stage` via label.split("_") (exactly 3
tokens). Images resolve at Data/test/<id>/<id>.jpg.

Tasks (choose with --task):
  crop3_disease : crop 3, 5 classes {3_00_0, 3_b3_1, 3_b6_1, 3_b7_1, 3_b8_1}  (DEFAULT)
  healthy       : binary healthy (disease=='00') vs diseased, all rows
  species       : predict crop (6 classes), all rows  [sanity only; shape shortcut]
  stage         : ordinal 3_a9 chain {3_a9_1, 3_a9_2, 3_a9_3} -> {0,1,2}

Writes splits/split_seed{N}.json with per-split ids + integer labels, the class
list, and the label map, so every downstream model consumes identical splits.

This script is DONE. Run it; do not rewrite it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict

from paths import DATA_ROOT
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
IMG_ROOT = os.path.join(DATA_ROOT, "test")  # Data/test/<id>/<id>.jpg

TASK_CLASSES = {
    "crop3_disease": ["3_00_0", "3_b3_1", "3_b6_1", "3_b7_1", "3_b8_1"],
    "stage": ["3_a9_1", "3_a9_2", "3_a9_3"],
}


def parse_label(label: str):
    parts = label.split("_")
    assert len(parts) == 3, f"label {label!r} does not split into 3 tokens"
    crop, disease, stage = parts
    return crop, disease, stage


def image_path(image_id: str) -> str:
    return os.path.join(IMG_ROOT, image_id, f"{image_id}.jpg")


def load_rows():
    with open(TEST_CSV, newline="") as f:
        return list(csv.DictReader(f))


def target_for_task(task: str, label: str, class_to_idx: dict):
    """Return integer target for this row under the task, or None to skip."""
    crop, disease, stage = parse_label(label)
    if task in ("crop3_disease", "stage"):
        return class_to_idx.get(label)
    if task == "healthy":
        return 0 if disease == "00" else 1
    if task == "species":
        return int(crop) - 1  # crops are 1..6 -> 0..5
    raise ValueError(f"unknown task {task}")


def build_class_index(task: str, rows):
    if task in TASK_CLASSES:
        classes = TASK_CLASSES[task]
    elif task == "healthy":
        classes = ["healthy", "diseased"]
    elif task == "species":
        crops = sorted({parse_label(r["label"])[0] for r in rows}, key=int)
        classes = [f"crop_{c}" for c in crops]
    else:
        raise ValueError(f"unknown task {task}")
    return classes, {c: i for i, c in enumerate(classes)}


def stratified_split(ids_by_class, seed, ratios=(0.6, 0.2, 0.2)):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls, ids in sorted(ids_by_class.items()):
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        # guarantee at least 1 in val/test when the class is tiny
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(1, min(n_va, n - n_tr - 1))
        train += ids[:n_tr]
        val += ids[n_tr:n_tr + n_va]
        test += ids[n_tr + n_va:]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="crop3_disease",
                    choices=["crop3_disease", "healthy", "species", "stage"])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "splits"))
    ap.add_argument("--check-images", action="store_true",
                    help="verify every selected image file exists (slower)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = load_rows()
    classes, class_to_idx = build_class_index(args.task, rows)

    # collect ids per integer target (skip rows not in task)
    ids_by_class = defaultdict(list)
    label_of = {}
    for r in rows:
        image_id, label = r["image"], r["label"]
        tgt = target_for_task(args.task, label, class_to_idx)
        if tgt is None:
            continue
        ids_by_class[tgt].append(image_id)
        label_of[image_id] = tgt

    counts = {classes[k]: len(v) for k, v in sorted(ids_by_class.items())}
    print(f"task={args.task}  classes={classes}")
    print(f"class counts: {counts}  total={sum(counts.values())}")

    if args.check_images:
        missing = [i for i in label_of if not os.path.isfile(image_path(i))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:5]}")
        print(f"image check OK: all {len(label_of)} images resolve")

    for seed in args.split_seeds:
        train, val, test = stratified_split(ids_by_class, seed)
        payload = {
            "task": args.task,
            "seed": seed,
            "classes": classes,
            "label_map": {i: label_of[i] for i in train + val + test},
            "image_path_template": "Data/test/<id>/<id>.jpg",
            "train": sorted(train, key=int),
            "val": sorted(val, key=int),
            "test": sorted(test, key=int),
        }
        out = os.path.join(out_dir, f"split_seed{seed}.json")
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"seed {seed}: train={len(train)} val={len(val)} test={len(test)} -> {out}")


if __name__ == "__main__":
    main()
