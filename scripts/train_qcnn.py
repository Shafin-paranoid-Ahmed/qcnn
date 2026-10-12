"""Train the clean Phase-A Patch-QCNN and perform the required budget smoke."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from metrics import classification_metrics
from patch_encode import load_image_patches
from qcnn_model import build_qcnn

from paths import ROOT, DATA_ROOT


class SplitDataset(Dataset):
    def __init__(self, split, part, encoding):
        self.ids = [str(value) for value in split[part]]
        self.labels = [int(split["label_map"][image_id]) for image_id in self.ids]
        self.encoding = encoding

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        path = os.path.join(DATA_ROOT, "test", image_id, f"{image_id}.jpg")
        return load_image_patches(path, encoding=self.encoding), self.labels[index]


def evaluate(model, loader, classes):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for patches, target in loader:
            probabilities.append(torch.softmax(model(patches), dim=1).cpu().numpy())
            labels.append(target.numpy())
    return classification_metrics(np.concatenate(probabilities), np.concatenate(labels), classes)


def train_one_seed(args):
    tag = getattr(args, "tag", None) or f"L{args.reupload_L}_{args.encoding}"
    if any(character in tag for character in ("/", "\\")) or tag in ("", ".", ".."):
        raise ValueError("tag must be a non-empty filename component")
    args.tag = tag
    random.seed(args.split_seed); np.random.seed(args.split_seed); torch.manual_seed(args.split_seed)
    split_path = os.path.join(ROOT, "splits", f"split_seed{args.split_seed}.json")
    with open(split_path) as handle:
        split = json.load(handle)
    train = SplitDataset(split, "train", args.encoding)
    val = SplitDataset(split, "val", args.encoding)
    test = SplitDataset(split, "test", args.encoding)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    attention = bool(getattr(args, "attention", False))
    model = build_qcnn(len(split["classes"]), args.d_conv, reupload_L=args.reupload_L,
                       encoding=args.encoding, attention=attention)
    expected_params = 138 if attention else 133
    if args.encoding == "amplitude10_mag" and args.reupload_L == 1 and model.trainable_parameter_count != expected_params:
        raise AssertionError(f"QCNN-mag parameter count changed: {model.trainable_parameter_count}")
    counts = np.bincount(train.labels, minlength=len(split["classes"]))
    weights = torch.tensor(len(train) / (len(counts) * counts), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    results_dir, checkpoint_dir = os.path.join(ROOT, "results"), os.path.join(ROOT, "checkpoints")
    os.makedirs(results_dir, exist_ok=True); os.makedirs(checkpoint_dir, exist_ok=True)
    artifact = f"qcnn_{tag}"
    model_label = "QCNN-attn" if attention else ("QCNN-mag" if args.encoding == "amplitude10_mag" else "QCNN")
    best_f1, history = -1.0, []
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter(); model.train(); losses = []; grad_norms = []
        for patches, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(patches), target, weight=weights)
            loss.backward()
            grad_norms.append(float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None))))
            optimizer.step(); losses.append(float(loss.detach()))
        val_metrics = evaluate(model, val_loader, split["classes"])
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "grad_norm": float(np.mean(grad_norms)), "wall_seconds": elapsed, **val_metrics}
        row["per_class_recall"] = json.dumps(row["per_class_recall"]); history.append(row)
        print(f"epoch={epoch} loss={row['train_loss']:.4f} val_macro_f1={row['macro_f1']:.4f} wall_s={elapsed:.1f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, patience_left = val_metrics["macro_f1"], args.patience
            torch.save({"state_dict": model.state_dict(), "classes": split["classes"], "args": vars(args)}, os.path.join(checkpoint_dir, f"{artifact}_seed{args.split_seed}.pt"))
        else:
            patience_left -= 1
        scheduler.step()
        if patience_left <= 0 and not args.smoke:
            break
    history_path = os.path.join(results_dir, f"{artifact}_seed{args.split_seed}_history.csv")
    with open(history_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    outcome = {"model": model_label, "seed": args.split_seed, "task": split["task"], "encoding": args.encoding, "d_conv": args.d_conv, "reupload_L": args.reupload_L, "params": model.trainable_parameter_count, "calibration_source": "genuine_softmax_after_class_weighted_training", "best_val_macro_f1": best_f1, "history": history_path}
    if not args.smoke:
        checkpoint_path = os.path.join(checkpoint_dir, f"{artifact}_seed{args.split_seed}.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        outcome["test"] = evaluate(model, test_loader, split["classes"])
    with open(os.path.join(results_dir, f"{artifact}_seed{args.split_seed}_metrics.json"), "w") as handle:
        json.dump(outcome, handle, indent=2)
    return outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--d-conv", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--reupload-L", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--encoding", choices=["amplitude10", "amplitude10_mag", "channel8"], default="amplitude10")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tag", help="artifact tag (default: L{reupload_L}_{encoding})")
    parser.add_argument("--attention", action="store_true", help="enable the pre-registered 5-param patch gate")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke: args.epochs = 1
    outcome = train_one_seed(args)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
