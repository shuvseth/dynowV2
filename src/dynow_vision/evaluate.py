"""Evaluate a trained checkpoint on the held-out test split.

    python -m dynow_vision.evaluate --checkpoint outputs/runs/<run>/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .dataset import DEFAULT_MANIFEST, WinterDataset, check_split_integrity, read_manifest
from .model import load_checkpoint, select_device
from .paths import repo_path
from .preprocessing import eval_transform, load_rgb

PRELIMINARY_MIN_TOTAL = 100
PRELIMINARY_MIN_PER_CLASS = 30


@torch.no_grad()
def predict_split(model, ds, device, batch_size=16):
    model.eval()
    names, y_true, y_pred, scores = [], [], [], []
    for x, y, fn in DataLoader(ds, batch_size):
        probs = torch.softmax(model(x.to(device)), 1).cpu()
        y_true += y.tolist(); y_pred += probs.argmax(1).tolist()
        scores += probs.max(1).values.tolist(); names += list(fn)
    return names, np.array(y_true), np.array(y_pred), np.array(scores)


def compute_metrics(y_true, y_pred, classes) -> dict:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    n = len(y_true)
    support = {c: int(s[i]) for i, c in enumerate(classes)}
    preliminary = n < PRELIMINARY_MIN_TOTAL or min(support.values()) < PRELIMINARY_MIN_PER_CLASS
    return {
        "n_test": int(n),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(classes)},
        "macro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": list(classes),
        "per_class_sample_counts": support,
        "preliminary": bool(preliminary),
        "note": ("PRELIMINARY: test set has fewer than "
                 f"{PRELIMINARY_MIN_TOTAL} images or fewer than {PRELIMINARY_MIN_PER_CLASS} per class; "
                 "these numbers have wide uncertainty and are not evidence of generalizable performance."
                 if preliminary else "Test set meets the minimum size heuristics; still a single split."),
    }


def plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Test confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_examples(rows, raw, classes, path, k=4):
    """Grid of up to k correct and k incorrect test predictions."""
    correct = [r for r in rows if r["correct"]][:k]
    wrong = [r for r in rows if not r["correct"]][:k]
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 6.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for row_i, (title, group) in enumerate((("correct", correct), ("incorrect", wrong))):
        for j, r in enumerate(group):
            axes[row_i][j].imshow(load_rgb(raw / r["filename"]))
            axes[row_i][j].set_title(f"{title}\ntrue={r['true']} pred={r['pred']} ({r['score']:.2f})", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def evaluate(checkpoint, manifest=DEFAULT_MANIFEST, split="test", out_dir=None, device="auto") -> dict:
    dev = select_device(device)
    model, ck = load_checkpoint(checkpoint, dev)
    cfg, classes = ck["config"], tuple(ck["classes"])
    rows = read_manifest(manifest)
    check_split_integrity(rows)
    ds = WinterDataset(rows, split, eval_transform(cfg["input_size"]), cfg["raw_dir"], classes)
    if len(ds) == 0:
        raise SystemExit(f"no labeled images in split {split!r}; cannot evaluate")
    names, y, yp, sc = predict_split(model, ds, dev)
    out = Path(out_dir) if out_dir else repo_path("outputs/evaluation") / f"{ck['run_name']}_{Path(checkpoint).stem}_{split}"
    out.mkdir(parents=True, exist_ok=True)

    m = compute_metrics(y, yp, classes)
    m.update({"split": split, "checkpoint": str(checkpoint), "checkpoint_sha256": ck["checkpoint_sha256"],
              "run_name": ck["run_name"], "trained_epoch_of_checkpoint": ck["epoch"],
              "model_scores_are_calibrated": False})
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    pred_rows = [{"filename": n, "true": classes[t], "pred": classes[p], "score": float(s), "correct": bool(t == p)}
                 for n, t, p, s in zip(names, y, yp, sc)]
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0])); w.writeheader(); w.writerows(pred_rows)
    plot_confusion(m["confusion_matrix"], list(classes), out / "confusion_matrix.png")
    plot_examples(pred_rows, repo_path(cfg["raw_dir"]), classes, out / "examples.png")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dynow_vision.evaluate", description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out-dir")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    m = evaluate(a.checkpoint, a.manifest, a.split, a.out_dir, a.device)
    print(json.dumps({k: m[k] for k in ("split", "n_test", "accuracy", "per_class", "confusion_matrix", "note")}, indent=2))


if __name__ == "__main__":
    main()
