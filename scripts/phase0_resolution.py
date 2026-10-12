"""Phase 0 — resolution sanity (spec Section 4).

Confirm disease-relevant information actually needs resolution, so the pilot is
not secretly a downscaled toy. Train a simple classical classifier (logistic
regression on flattened pixels) on the chosen task at several resolutions and
report Macro-F1. Expect Macro-F1 to rise with resolution if fine texture matters.

Needs no quantum libraries. This script is DONE; run it after build_splits.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from paths import DATA_ROOT
IMG_ROOT = os.path.join(DATA_ROOT, "test")


def image_path(image_id: str) -> str:
    return os.path.join(IMG_ROOT, image_id, f"{image_id}.jpg")


def load_split(path):
    with open(path) as f:
        return json.load(f)


def load_matrix(ids, label_map, res, cap_per_class, seed):
    """Load and flatten images at res x res; optionally cap ids per class."""
    rng = np.random.RandomState(seed)
    if cap_per_class:
        by_cls = {}
        for i in ids:
            by_cls.setdefault(label_map[i], []).append(i)
        kept = []
        for cls, group in by_cls.items():
            group = list(group)
            rng.shuffle(group)
            kept += group[:cap_per_class]
        ids = kept
    X, y = [], []
    for i in ids:
        img = Image.open(image_path(i)).convert("RGB").resize((res, res))
        X.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
        y.append(label_map[i])
    return np.asarray(X), np.asarray(y), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(os.path.dirname(__file__), "..", "splits", "split_seed0.json"))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[28, 56, 112, 224])
    ap.add_argument("--cap-per-class", type=int, default=150,
                    help="cap ids per class for speed; 0 = use all")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "phase0_resolution.csv"))
    args = ap.parse_args()

    split = load_split(args.split)
    # label_map keys are strings in JSON; normalize
    label_map = {str(k): int(v) for k, v in split["label_map"].items()}
    train_ids = [str(i) for i in split["train"]]
    val_ids = [str(i) for i in split["val"]]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for res in args.resolutions:
        Xtr, ytr, ntr = load_matrix(train_ids, label_map, res, args.cap_per_class, split["seed"])
        Xva, yva, nva = load_matrix(val_ids, label_map, res, 0, split["seed"])
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(scaler.transform(Xtr), ytr)
        pred = clf.predict(scaler.transform(Xva))
        f1 = f1_score(yva, pred, average="macro")
        print(f"res={res:>3}  train_n={ntr}  val_n={nva}  val_macroF1={f1:.4f}")
        rows.append({"task": split["task"], "resolution": res,
                     "train_n": ntr, "val_n": nva, "val_macro_f1": round(float(f1), 4)})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
