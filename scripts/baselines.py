"""Mandatory clean Phase-A controls: parameter-matched B-PCNN and B-QUAD."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from metrics import classification_metrics
from patch_encode import extract_patches, per_image_angle_normalize
from qcnn_model import PatchAttentionGate

from paths import ROOT, DATA_ROOT


def patch_pixels(image_id):
    path = os.path.join(DATA_ROOT, "test", image_id, f"{image_id}.jpg")
    with Image.open(path) as image:
        pixels = torch.from_numpy(np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32).copy())
    patches = extract_patches(per_image_angle_normalize(pixels), patch=16)
    return patches.reshape(196, 16, 16, 3).permute(0, 3, 1, 2).float()


def patch_channel_means(image_id):
    return patch_pixels(image_id).mean(dim=(2, 3))


class PatchMeanDataset(Dataset):
    def __init__(self, split, part):
        self.ids = [str(value) for value in split[part]]
        self.labels = [int(split["label_map"][image_id]) for image_id in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return patch_channel_means(self.ids[index]), self.labels[index]


class RawPatchDataset(PatchMeanDataset):
    def __getitem__(self, index):
        return patch_pixels(self.ids[index]), self.labels[index]


class ParameterMatchedPCNN(nn.Module):
    """Spatial shared patch filter plus the same 14x14 -> 7x7 -> global hierarchy."""
    def __init__(self, n_classes, attention=False):
        super().__init__()
        self.depthwise = nn.Conv2d(3, 3, kernel_size=2, groups=3, bias=False)
        self.pointwise = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.patch_mix = nn.Linear(4, 4, bias=False)
        self.attention = PatchAttentionGate(4) if attention else None
        self.pool1 = nn.Linear(4, 4, bias=False)
        self.pool2 = nn.Linear(4, 8, bias=False)
        self.readout = nn.Linear(8, n_classes)

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, patches, return_attention=False):
        batch = patches.shape[0]
        filtered = self.pointwise(self.depthwise(patches.reshape(-1, 3, 16, 16)))
        features = torch.tanh(self.patch_mix(filtered.mean(dim=(2, 3)))).reshape(batch, 196, 4)
        attention = None
        if self.attention is not None:
            features, attention = self.attention(features, return_attention=True)
        elif return_attention:
            raise ValueError("attention weights requested from a model with attention disabled")
        grid = features.reshape(batch, 14, 14, 4)
        cells = grid.reshape(batch, 7, 2, 7, 2, 4).mean(dim=(2, 4))
        level1 = torch.tanh(self.pool1(cells.reshape(batch, 49, 4)))
        logits = self.readout(torch.tanh(self.pool2(level1.mean(dim=1))))
        return (logits, attention) if return_attention else logits


class AttentionOnlyNull(nn.Module):
    """B-ATTN: attention over RGB patch means, global mean, linear head."""
    def __init__(self, n_classes):
        super().__init__()
        self.attention = PatchAttentionGate(3)
        self.readout = nn.Linear(3, n_classes)

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, patch_means, return_attention=False):
        gated, attention = self.attention(patch_means, return_attention=True)
        logits = self.readout(gated.mean(dim=1))
        return (logits, attention) if return_attention else logits


def evaluate_torch(model, loader, classes):
    model.eval(); probabilities, labels = [], []
    with torch.no_grad():
        for inputs, target in loader:
            probabilities.append(torch.softmax(model(inputs), dim=1).numpy())
            labels.append(target.numpy())
    return classification_metrics(np.concatenate(probabilities), np.concatenate(labels), classes)


def train_bpcnn(split, seed, epochs=100, patience=15, batch_size=4, lr=1e-2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train, val, test = (RawPatchDataset(split, part) for part in ("train", "val", "test"))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    model = ParameterMatchedPCNN(len(split["classes"]))
    if model.trainable_parameter_count != 133:
        raise AssertionError(f"B-PCNN has {model.trainable_parameter_count} params; QCNN has 133")
    counts = np.bincount(train.labels, minlength=len(split["classes"]))
    weights = torch.tensor(len(train) / (len(counts) * counts), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_f1, best_state, patience_left = -1.0, None, patience
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter(); model.train(); losses = []
        for inputs, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), target, weight=weights)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val_metrics = evaluate_torch(model, val_loader, split["classes"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "wall_seconds": time.perf_counter() - started,
                        "val_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        scheduler.step()
        if patience_left <= 0:
            break
    model.load_state_dict(best_state)
    checkpoint_dir = os.path.join(ROOT, "checkpoints"); result_dir = os.path.join(ROOT, "results")
    os.makedirs(checkpoint_dir, exist_ok=True); os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"bpcnn_seed{seed}.pt")
    torch.save({"state_dict": best_state, "classes": split["classes"], "seed": seed}, checkpoint_path)
    history_path = os.path.join(result_dir, f"bpcnn_seed{seed}_history.csv")
    with open(history_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    return {"model": "B-PCNN", "seed": seed, "params": model.trainable_parameter_count,
            "calibration_source": "genuine_softmax_after_class_weighted_training",
            "best_val_macro_f1": best_f1, "history": history_path,
            "checkpoint": checkpoint_path, "test": evaluate_torch(model, test_loader, split["classes"])}


def pooled_quad_features(dataset):
    features, labels = [], []
    for index in range(len(dataset)):
        patch_means, target = dataset[index]
        grid = patch_means.reshape(14, 14, 3)
        pooled = grid.reshape(7, 2, 7, 2, 3).mean(dim=(1, 3)).reshape(-1)
        features.append(pooled.numpy()); labels.append(target)
    return np.asarray(features), np.asarray(labels)


def train_bquad(split, seed):
    train, val, test = (PatchMeanDataset(split, part) for part in ("train", "val", "test"))
    x_train, y_train = pooled_quad_features(train)
    x_val, y_val = pooled_quad_features(val)
    x_test, y_test = pooled_quad_features(test)
    model = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                          LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed))
    calibration_model = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                      LogisticRegression(max_iter=1000, class_weight=None, solver="liblinear", random_state=seed))
    model.fit(x_train, y_train); calibration_model.fit(x_train, y_train)
    val_metrics = classification_metrics(model.predict_proba(x_val), y_val, split["classes"], calibration_model.predict_proba(x_val))
    test_metrics = classification_metrics(model.predict_proba(x_test), y_test, split["classes"], calibration_model.predict_proba(x_test))
    return {"model": "B-QUAD", "seed": seed, "params": None,
            "calibration_source": "second_unweighted_logistic_model",
            "best_val_macro_f1": val_metrics["macro_f1"], "test": test_metrics}


def run_baselines(split_seed, models=("bpcnn", "bquad"), epochs=100):
    with open(os.path.join(ROOT, "splits", f"split_seed{split_seed}.json")) as handle:
        split = json.load(handle)
    results = []
    if "bpcnn" in models:
        results.append(train_bpcnn(split, split_seed, epochs=epochs))
    if "bquad" in models:
        results.append(train_bquad(split, split_seed))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"baselines_seed{split_seed}_metrics.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", choices=["bpcnn", "bquad"], default=["bpcnn", "bquad"])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_baselines(args.split_seed, args.models, args.epochs), indent=2))


if __name__ == "__main__":
    main()
