"""Phase A3 attention arms and preregistered saliency exports."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from baselines import (AttentionOnlyNull, ParameterMatchedPCNN, PatchMeanDataset,
                       RawPatchDataset, evaluate_torch)
from metrics import classification_metrics
from train_qcnn import ROOT, SplitDataset, evaluate
from paths import DATA_ROOT
from qcnn_model import PatchQCNN, build_qcnn


ARM_TAGS = {
    "qcnn_attn": "qcnn_attn_L1_mag",
    "bpcnn_attn": "bpcnn_attn",
    "battn": "battn",
}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_split(seed):
    with open(os.path.join(ROOT, "splits", f"split_seed{seed}.json")) as handle:
        return json.load(handle)


def build_arm(arm, n_classes):
    if arm == "qcnn_attn":
        model = build_qcnn(n_classes, d_conv=2, reupload_L=1,
                           encoding="amplitude10_mag", attention=True)
        expected = 138
    elif arm == "bpcnn_attn":
        model, expected = ParameterMatchedPCNN(n_classes, attention=True), 138
    elif arm == "battn":
        model, expected = AttentionOnlyNull(n_classes), 24
    else:
        raise ValueError(f"unknown Phase A3 arm {arm!r}")
    if model.trainable_parameter_count != expected:
        raise AssertionError(f"{arm} must have {expected} trainable parameters, got {model.trainable_parameter_count}")
    return model


def datasets_for(arm, split):
    if arm == "qcnn_attn":
        return tuple(SplitDataset(split, part, "amplitude10_mag") for part in ("train", "val", "test"))
    dataset = RawPatchDataset if arm == "bpcnn_attn" else PatchMeanDataset
    return tuple(dataset(split, part) for part in ("train", "val", "test"))


def evaluate_arm(model, loader, classes):
    return evaluate(model, loader, classes) if isinstance(model, PatchQCNN) else evaluate_torch(model, loader, classes)


def train_arm(arm, seed, epochs=100, patience=15, batch_size=4, lr=1e-2, smoke=False):
    seed_everything(seed)
    split = load_split(seed)
    train, val, test = datasets_for(arm, split)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_arm(arm, len(split["classes"]))
    counts = np.bincount(train.labels, minlength=len(split["classes"]))
    weights = torch.tensor(len(train) / (len(counts) * counts), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 1 if smoke else epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_f1, patience_left, history = -1.0, patience, []
    tag = ARM_TAGS[arm] + ("_smoke" if smoke else "")
    checkpoint_dir, result_dir = os.path.join(ROOT, "checkpoints"), os.path.join(ROOT, "results")
    os.makedirs(checkpoint_dir, exist_ok=True); os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{tag}_seed{seed}.pt")
    for epoch in range(1, epochs + 1):
        started = time.perf_counter(); model.train(); losses, grad_norms = [], []
        for inputs, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), target, weight=weights)
            loss.backward()
            grad_norms.append(float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None))))
            optimizer.step(); losses.append(float(loss.detach()))
        val_metrics = evaluate_arm(model, val_loader, split["classes"])
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "grad_norm": float(np.mean(grad_norms)), "wall_seconds": elapsed,
                        **{k: v for k, v in val_metrics.items() if k != "per_class_recall"},
                        "per_class_recall": json.dumps(val_metrics["per_class_recall"])})
        print(f"arm={arm} seed={seed} epoch={epoch} val_macro_f1={val_metrics['macro_f1']:.4f} wall_s={elapsed:.1f}", flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, patience_left = val_metrics["macro_f1"], patience
            torch.save({"state_dict": model.state_dict(), "classes": split["classes"],
                        "arm": arm, "seed": seed}, checkpoint_path)
        else:
            patience_left -= 1
        scheduler.step()
        if patience_left <= 0 and not smoke:
            break
    history_path = os.path.join(result_dir, f"{tag}_seed{seed}_history.csv")
    with open(history_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    outcome = {"model": {"qcnn_attn": "QCNN-attn", "bpcnn_attn": "B-PCNN-attn", "battn": "B-ATTN"}[arm],
               "tag": tag, "seed": seed, "params": model.trainable_parameter_count,
               "task": split["task"],
               "encoding": "amplitude10_mag" if arm == "qcnn_attn" else "",
               "d_conv": 2 if arm == "qcnn_attn" else "",
               "reupload_L": 1 if arm == "qcnn_attn" else "",
               "best_val_macro_f1": best_f1, "history": history_path,
               "checkpoint": checkpoint_path,
               "calibration_source": "genuine_softmax_after_class_weighted_training"}
    if not smoke:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"])
        outcome["test"] = evaluate_arm(model, test_loader, split["classes"])
    with open(os.path.join(result_dir, f"{tag}_seed{seed}_metrics.json"), "w") as handle:
        json.dump(outcome, handle, indent=2)
    return outcome


def _saliency_candidates(model, dataset, classes):
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    rows = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            logits, alpha = model(inputs, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(1)
            for j in range(len(target)):
                rows.append({"index": offset + j, "id": dataset.ids[offset + j],
                             "true": int(target[j]), "pred": int(pred[j]),
                             "confidence": float(probs[j, pred[j]]),
                             "alpha": alpha[j].cpu().numpy()})
            offset += len(target)
    chosen = []
    for class_index, class_name in enumerate(classes):
        correct = sorted((r for r in rows if r["true"] == class_index and r["pred"] == class_index),
                         key=lambda r: r["confidence"], reverse=True)[:3]
        wrong = sorted((r for r in rows if r["true"] == class_index and r["pred"] != class_index),
                       key=lambda r: r["confidence"], reverse=True)[:2]
        for rank, row in enumerate(correct, 1): chosen.append((class_name, "correct", rank, row))
        for rank, row in enumerate(wrong, 1): chosen.append((class_name, "misclassified", rank, row))
    return chosen


def export_saliency(arm, seed=0):
    split = load_split(seed); dataset = datasets_for(arm, split)[2]
    model = build_arm(arm, len(split["classes"]))
    checkpoint = os.path.join(ROOT, "checkpoints", f"{ARM_TAGS[arm]}_seed{seed}.pt")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    saliency_dir, figure_dir = os.path.join(ROOT, "results", "saliency"), os.path.join(ROOT, "results", "figures")
    os.makedirs(saliency_dir, exist_ok=True); os.makedirs(figure_dir, exist_ok=True)
    index = []
    for class_name, outcome, rank, row in _saliency_candidates(model, dataset, split["classes"]):
        stem = f"{ARM_TAGS[arm]}_seed0_{class_name}_{outcome}{rank}_{row['id']}"
        csv_path, png_path = os.path.join(saliency_dir, stem + ".csv"), os.path.join(figure_dir, stem + ".png")
        np.savetxt(csv_path, row["alpha"].reshape(14, 14), delimiter=",", fmt="%.10g")
        image_path = os.path.join(DATA_ROOT, "test", row["id"], f"{row['id']}.jpg")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").resize((224, 224))
        fig, axis = plt.subplots(figsize=(5, 5)); axis.imshow(rgb)
        axis.imshow(row["alpha"].reshape(14, 14), cmap="jet", alpha=.45,
                    interpolation="bilinear", extent=(0, 224, 224, 0))
        axis.axis("off"); fig.tight_layout(pad=0); fig.savefig(png_path, dpi=160, bbox_inches="tight", pad_inches=0); plt.close(fig)
        index.append({"arm": arm, "class": class_name, "outcome": outcome, "rank": rank,
                      "image_id": row["id"], "predicted_class": split["classes"][row["pred"]],
                      "confidence": row["confidence"], "alpha_csv": os.path.relpath(csv_path, ROOT),
                      "overlay_png": os.path.relpath(png_path, ROOT)})
    index_path = os.path.join(ROOT, "results", "saliency", f"{ARM_TAGS[arm]}_seed{seed}_index.csv")
    with open(index_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index[0].keys()); writer.writeheader(); writer.writerows(index)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(ARM_TAGS), required=True)
    parser.add_argument("--split-seed", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--saliency", action="store_true")
    args = parser.parse_args()
    if args.saliency:
        print(json.dumps(export_saliency(args.arm, args.split_seed), indent=2)); return
    print(json.dumps(train_arm(args.arm, args.split_seed, args.epochs, args.patience,
                               args.batch_size, args.lr, args.smoke), indent=2))


if __name__ == "__main__":
    main()
