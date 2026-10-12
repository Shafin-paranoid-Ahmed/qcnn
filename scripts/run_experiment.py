"""Run clean Phase A across seeds and aggregate QCNN/B-PCNN/B-QUAD results."""
from __future__ import annotations

import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np

from baselines import run_baselines
from train_qcnn import train_one_seed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")


def flatten_result(result):
    row = {key: value for key, value in result.items() if key not in ("test", "history")}
    for key, value in result.get("test", {}).items():
        row[f"test_{key}"] = json.dumps(value) if isinstance(value, dict) else value
    return row


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows):
    summaries = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        summary = {"model": model, "seeds": len(group)}
        for metric in ("test_accuracy", "test_macro_f1", "test_macro_recall", "test_ece_15bin", "test_brier", "test_nll"):
            values = [float(row[metric]) for row in group if metric in row]
            if values:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        recalls = [json.loads(row["test_per_class_recall"]) for row in group if row.get("test_per_class_recall")]
        if recalls:
            for class_name in recalls[0]:
                values = [float(item[class_name]) for item in recalls]
                summary[f"test_recall_{class_name}_mean"] = float(np.mean(values))
                summary[f"test_recall_{class_name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_report(summary):
    by_model = {row["model"]: row for row in summary}
    lines = ["# Q0 Clean Phase-A and Magnitude-Ablation Report", "", "True-label `crop3_disease` splits from `Data/test.csv` only.", "",
             "| Model | Macro-F1 | Accuracy | ECE | Brier | NLL | Macro recall |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in ("QCNN", "QCNN-mag", "B-PCNN", "B-QUAD"):
        row = by_model.get(model)
        if row:
            lines.append(f"| {model} | {row['test_macro_f1_mean']:.4f} +/- {row['test_macro_f1_std']:.4f} | {row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f} | {row['test_ece_15bin_mean']:.4f} +/- {row['test_ece_15bin_std']:.4f} | {row['test_brier_mean']:.4f} +/- {row['test_brier_std']:.4f} | {row['test_nll_mean']:.4f} +/- {row['test_nll_std']:.4f} | {row['test_macro_recall_mean']:.4f} +/- {row['test_macro_recall_std']:.4f} |")
    class_names = sorted(key.removeprefix("test_recall_").removesuffix("_mean") for row in summary for key in row if key.startswith("test_recall_") and key.endswith("_mean"))
    lines += ["", "## Per-class recall", "", "| Model | " + " | ".join(class_names) + " |", "| --- | " + " | ".join(["---:"] * len(class_names)) + " |"]
    for model in ("QCNN", "QCNN-mag", "B-PCNN", "B-QUAD"):
        row = by_model.get(model)
        if row:
            cells = [f"{row[f'test_recall_{name}_mean']:.4f} +/- {row[f'test_recall_{name}_std']:.4f}" for name in class_names]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
    with open(os.path.join(RESULTS, "QCNN_REPORT.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def write_studies(rows, summary):
    qcnn_rows = [row for row in rows if row["model"] == "QCNN"]
    scalability = [{"seed": row["seed"], "patch_size": 16, "qubits": 10,
                    "d_conv": row["d_conv"], "reupload_L": row["reupload_L"],
                    "params": row["params"], "test_macro_f1": row.get("test_macro_f1")}
                   for row in qcnn_rows]
    write_csv(os.path.join(RESULTS, "scalability.csv"), scalability)
    by_model = {row["model"]: row for row in summary}
    qcnn, quad = by_model.get("QCNN"), by_model.get("B-QUAD")
    text = ["# Function-Class Study", "", "B-QUAD is the explicit degree-2 null on identical fixed pooled raw-pixel summaries.", ""]
    if qcnn and quad:
        delta = qcnn["test_macro_f1_mean"] - quad["test_macro_f1_mean"]
        text += [f"QCNN mean test Macro-F1: {qcnn['test_macro_f1_mean']:.6f}.",
                 f"B-QUAD mean test Macro-F1: {quad['test_macro_f1_mean']:.6f}.",
                 f"QCNN minus B-QUAD: {delta:.6f}.", "",
                 "This L=1 run alone cannot establish richer-than-quadratic behavior; L>1 and B-POLY2 ablations remain required for that claim."]
    with open(os.path.join(RESULTS, "function_class.md"), "w") as handle:
        handle.write("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A"], default="A")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--encoding", choices=["amplitude10", "amplitude10_mag", "channel8"], default="amplitude10")
    parser.add_argument("--d-conv", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--reupload-L", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for seed in args.split_seeds:
        if args.encoding == "amplitude10_mag":
            for name in (f"qcnn_seed{seed}_metrics.json", f"baselines_seed{seed}_metrics.json"):
                with open(os.path.join(RESULTS, name)) as handle:
                    loaded = json.load(handle)
                rows.extend(flatten_result(item) for item in loaded) if isinstance(loaded, list) else rows.append(flatten_result(loaded))
            mag_path = os.path.join(RESULTS, f"qcnn_mag_seed{seed}_metrics.json")
            if args.aggregate_only or os.path.exists(mag_path):
                with open(mag_path) as handle:
                    rows.append(flatten_result(json.load(handle)))
            else:
                print(f"=== seed {seed}: QCNN-mag ===", flush=True)
                qargs = SimpleNamespace(split_seed=seed, d_conv=args.d_conv, reupload_L=args.reupload_L,
                                        encoding=args.encoding, epochs=args.epochs, patience=args.patience,
                                        batch_size=args.batch_size, lr=1e-2, smoke=False)
                rows.append(flatten_result(train_one_seed(qargs)))
            write_csv(os.path.join(RESULTS, "all_runs.csv"), rows)
            continue
        if args.aggregate_only:
            with open(os.path.join(RESULTS, f"qcnn_seed{seed}_metrics.json")) as handle:
                rows.append(flatten_result(json.load(handle)))
            with open(os.path.join(RESULTS, f"baselines_seed{seed}_metrics.json")) as handle:
                rows.extend(flatten_result(result) for result in json.load(handle))
        else:
            print(f"=== seed {seed}: QCNN ===", flush=True)
            qargs = SimpleNamespace(split_seed=seed, d_conv=args.d_conv, reupload_L=args.reupload_L,
                                    encoding=args.encoding, epochs=args.epochs, patience=args.patience,
                                    batch_size=args.batch_size, lr=1e-2, smoke=False)
            rows.append(flatten_result(train_one_seed(qargs)))
            print(f"=== seed {seed}: B-PCNN + B-QUAD ===", flush=True)
            rows.extend(flatten_result(result) for result in run_baselines(seed, epochs=args.baseline_epochs))
        write_csv(os.path.join(RESULTS, "all_runs.csv"), rows)
    summary = aggregate(rows)
    write_csv(os.path.join(RESULTS, "summary.csv"), summary)
    write_report(summary); write_studies(rows, summary)
    if args.encoding == "amplitude10_mag":
        print(json.dumps(summary, indent=2), flush=True)
        return
    by_model = {row["model"]: row for row in summary}
    qcnn, pcnn = by_model["QCNN"], by_model["B-PCNN"]
    gap = qcnn["test_macro_f1_mean"] - pcnn["test_macro_f1_mean"]
    verdict = "competitive-clean" if gap >= -0.03 else "do-not-expand"
    with open(os.path.join(ROOT, "RESULTS_LOG.md"), "a") as handle:
        handle.write(f"\n## Phase-A clean go/no-go\n\nQCNN minus B-PCNN test Macro-F1: {gap:.6f}. Verdict: **{verdict}**.\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
