"""Assemble the preregistered Phase A3 run table and report from frozen artifacts."""
from __future__ import annotations
import csv
import json
import os
from datetime import date

import numpy as np

from attention_experiment import ARM_TAGS
from train_qcnn import ROOT

METRICS = ("macro_f1", "accuracy", "ece_15bin", "brier", "nll", "macro_recall")


def load_all_runs():
    with open(os.path.join(ROOT, "results", "all_runs.csv"), newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def load_new_rows(fields):
    rows = []
    for arm, tag in ARM_TAGS.items():
        for seed in (0, 1, 2):
            path = os.path.join(ROOT, "results", f"{tag}_seed{seed}_metrics.json")
            with open(path) as handle: outcome = json.load(handle)
            row = {field: outcome.get(field, "") for field in fields}
            for key in METRICS:
                row[f"test_{key}"] = outcome["test"][key]
            row["test_per_class_recall"] = json.dumps(outcome["test"]["per_class_recall"])
            row["tag"] = tag
            rows.append(row)
    return rows


def load_anchor_rows(fields, source):
    rows = []
    for row in source:
        model = row["model"]
        is_mag = model == "QCNN-mag" and row.get("encoding") == "amplitude10_mag" and row.get("reupload_L") == "1"
        if not (is_mag or model in ("B-PCNN", "B-QUAD")): continue
        copied = {field: row.get(field, "") for field in fields}
        copied["seed"] = int(row["seed"])
        copied["tag"] = "anchor_reused"
        rows.append(copied)
    expected = {(m, s) for m in ("QCNN-mag", "B-PCNN", "B-QUAD") for s in (0, 1, 2)}
    found = {(r["model"], r["seed"]) for r in rows}
    if found != expected: raise RuntimeError(f"anchor set incomplete: missing={sorted(expected-found)} extra={sorted(found-expected)}")
    return rows


def fmt(values):
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"


def main():
    fields, source = load_all_runs()
    rows = load_new_rows(fields) + load_anchor_rows(fields, source)
    output = os.path.join(ROOT, "results", "attention_runs.csv")
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["tag"]); writer.writeheader(); writer.writerows(rows)
    groups = {}
    for row in rows: groups.setdefault(row["model"], []).append(row)
    order = ("QCNN-attn", "B-PCNN-attn", "B-ATTN", "QCNN-mag", "B-PCNN", "B-QUAD")
    lines = ["# Phase A3 Attention Report", "", f"Generated: {date.today().isoformat()}", "",
             "All values are mean ± sample standard deviation across seeds 0/1/2. Test data were evaluated once per seed after validation-only checkpoint selection.", "",
             "| Arm | Params | Macro-F1 | Accuracy | ECE | Brier | NLL | Macro recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in order:
        arm_rows = groups[model]
        params = arm_rows[0]["params"] or "n.a."
        lines.append("| " + " | ".join([model, str(params)] +
                     [fmt([float(r[f"test_{key}"]) for r in arm_rows]) for key in METRICS]) + " |")
    q = np.mean([float(r["test_macro_f1"]) for r in groups["QCNN-attn"]])
    anchor = np.mean([float(r["test_macro_f1"]) for r in groups["QCNN-mag"]])
    bpcnn = np.mean([float(r["test_macro_f1"]) for r in groups["B-PCNN-attn"]])
    battn = np.mean([float(r["test_macro_f1"]) for r in groups["B-ATTN"]])
    delta = q - anchor
    lines += ["", "## Pre-registered verdict", "",
              f"QCNN-attn − QCNN-mag mean Macro-F1: {delta:+.4f}. The ±0.03 expectation **{'held' if abs(delta) <= .03 else 'did not hold'}**.",
              f"QCNN-attn − B-PCNN-attn mean Macro-F1: {q-bpcnn:+.4f}. QCNN-attn **{'stayed ahead' if q > bpcnn else 'did not stay ahead'}**, against the pre-registered expectation.",
              f"B-ATTN − QCNN-attn mean Macro-F1: {battn-q:+.4f}; B-ATTN − B-PCNN-attn: {battn-bpcnn:+.4f}. B-ATTN **{'was below both gated hierarchical arms' if battn < min(q, bpcnn) else 'was not below both gated hierarchical arms'}**. No numeric threshold for 'NEAR' was pre-registered, so none is added post hoc.",
              "**B-ATTN 'NEAR' qualitative verdict: pending human review.** If B-ATTN is judged near QCNN-attn, record the pre-registered conclusion that attention, not the quantum hierarchy, carries the gain.",
              f"Level-2 trigger (QCNN-attn gain ≥ +0.02): **{'met, pending explicit team runway confirmation; do not run automatically' if delta >= .02 else 'not met; remains parked'}**.",
              "", "## Saliency figure index", ""]
    indices = []
    for tag in ARM_TAGS.values():
        path = os.path.join(ROOT, "results", "saliency", f"{tag}_seed0_index.csv")
        with open(path, newline="") as handle: indices.extend(csv.DictReader(handle))
    lines += ["| Arm | True class | Outcome | Rank | Image | Predicted | Confidence | Alpha CSV | Overlay PNG |",
              "| --- | --- | --- | ---: | --- | --- | ---: | --- | --- |"]
    for r in indices:
        lines.append(f"| {r['arm']} | {r['class']} | {r['outcome']} | {r['rank']} | {r['image_id']} | {r['predicted_class']} | {float(r['confidence']):.4f} | `{r['alpha_csv']}` | `{r['overlay_png']}` |")
    lines += ["", "## Saliency qualitative verdict", "",
              "**Pending human figure review.** Inspect the indexed overlays and record whether attention appears to concentrate on lesion/leaf regions rather than background for diseased classes. A contradictory result is the finding; do not change the pre-registered expectation.",
              "", "Saliency concentration is a qualitative, figure-based observation only; no lesion localization labels were used."]
    with open(os.path.join(ROOT, "results", "ATTENTION_REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__": main()
