"""Train the classification head on frozen DINOv2 features.

    python -m dynow_vision.train --config configs/phase1.yaml
    python -m dynow_vision.train --config configs/phase1.yaml --resume outputs/runs/<run>/last.pt

Only the train and val splits are read here; the test split is never loaded.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (SPLITS, WinterDataset, check_split_integrity, manifest_sha256,
                      read_manifest, validate_rows)
from .model import build_model, load_checkpoint, save_checkpoint, select_device
from .paths import repo_path
from .preprocessing import eval_transform, train_transform

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path("."),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def apply_repeats(ds: WinterDataset, repeat: dict | None) -> None:
    """Repeat train items whose filename starts with a key (each repeat is augmented differently).
    Used to give a tiny but important image group more weight; the true image count is unchanged."""
    if not repeat:
        return
    items = []
    for r in ds.items:
        n = next((v for k, v in repeat.items() if r["filename"].startswith(k)), 1)
        items.extend([r] * int(n))
    ds.items = items


def check_enough_data(rows: list[dict], cfg: dict) -> None:
    """Refuse to train on data that cannot support a meaningful run."""
    counts = {sp: {c: 0 for c in cfg["classes"]} for sp in SPLITS}
    for r in rows:
        if r["split"] in counts and r["label"] in cfg["classes"]:
            counts[r["split"]][r["label"]] += 1
    need = cfg["min_per_class"]
    short = [f"{sp}/{c}: have {counts[sp][c]}, need >= {need[sp]}"
             for sp in SPLITS for c in cfg["classes"] if counts[sp][c] < need[sp]]
    if short:
        raise SystemExit(
            "Not enough labeled data to train:\n  " + "\n  ".join(short) +
            "\nAdd and label more images (see README), run `python -m dynow_vision.dataset split`, "
            "then retry.")


@torch.no_grad()
def _eval_loop(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def train(cfg: dict, resume: str | None = None) -> Path:
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    rows = read_manifest(cfg["manifest"])
    problems = validate_rows(rows, cfg["raw_dir"])
    if problems:
        raise SystemExit("Manifest problems:\n  " + "\n  ".join(problems[:20]))
    check_split_integrity(rows)
    check_enough_data(rows, cfg)

    size, classes = cfg["input_size"], tuple(cfg["classes"])
    train_ds = WinterDataset(rows, "train", train_transform(size), cfg["raw_dir"], classes)
    apply_repeats(train_ds, cfg.get("repeat"))
    val_ds = WinterDataset(rows, "val", eval_transform(size), cfg["raw_dir"], classes)
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, generator=g,
                          num_workers=cfg["num_workers"])
    val_dl = DataLoader(val_ds, cfg["batch_size"], num_workers=cfg["num_workers"])

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    weight = None
    if cfg.get("class_weights") == "balanced":  # inverse-frequency weights from the TRAIN split only
        counts = torch.bincount(torch.tensor(train_ds.labels()), minlength=len(classes)).float()
        weight = (counts.sum() / (len(classes) * counts)).to(device)
        print("class weights:", dict(zip(classes, weight.tolist())))
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    start_epoch, history, best = 0, [], {"val_acc": -1.0, "val_loss": float("inf"), "epoch": -1}
    if resume:
        _, ck = load_checkpoint(resume, device)
        model.head.load_state_dict(ck["head_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch, history, best = ck["epoch"] + 1, ck["history"], ck["best"]
        run_dir = Path(resume).resolve().parent
        print(f"resuming {run_dir.name} from epoch {start_epoch}")
    else:
        name = cfg.get("run_name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = repo_path(cfg["output_root"]) / name
        if run_dir.exists():
            raise SystemExit(f"{run_dir} already exists; choose another run_name or use --resume")
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (run_dir / "environment.json").write_text(json.dumps({
            "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "git_commit": git_commit(),
            "manifest_sha256": manifest_sha256(rows),
            "n_train": len(train_ds), "n_val": len(val_ds),
            "trainable_params": sum(p.numel() for p in model.trainable_parameters()),
            "frozen_params": sum(p.numel() for p in model.backbone.parameters()),
            "backbone_source": "torch.hub facebookresearch/dinov2 (main branch), pretrained weights by Meta AI",
        }, indent=2))

    print(f"device={device} train={len(train_ds)} val={len(val_ds)} run={run_dir}")
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        val_loss, val_acc = _eval_loop(model, val_dl, loss_fn, device)
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "train_acc": correct / n,
               "val_loss": val_loss, "val_acc": val_acc, "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))

        improved = val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"])
        if improved:
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
        extra = {"epoch": epoch, "history": history, "best": best, "run_name": run_dir.name,
                 "manifest_sha256": manifest_sha256(rows), "git_commit": git_commit()}
        save_checkpoint(run_dir / "last.pt", model, cfg, extra, optimizer)
        if improved:
            save_checkpoint(run_dir / "best.pt", model, cfg, extra)

        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader(); w.writerows(history)
    (run_dir / "best.json").write_text(json.dumps(best, indent=2))
    print(f"best val_acc={best['val_acc']:.4f} at epoch {best['epoch']} -> {run_dir / 'best.pt'}")
    return run_dir


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(repo_path(path).read_text())
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dynow_vision.train", description=__doc__)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--resume", help="path to a last.pt to continue from")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-name", dest="run_name")
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {k: getattr(a, k) for k in ("epochs", "batch_size", "lr", "seed", "run_name", "device")})
    train(cfg, a.resume)


if __name__ == "__main__":
    main()
