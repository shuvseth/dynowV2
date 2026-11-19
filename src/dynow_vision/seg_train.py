"""Fine-tune the pretrained SegFormer to find the WALKABLE sidewalk (human-drawn masks only).

    python -m dynow_vision.seg_train train --run-name seg_v1
    python -m dynow_vision.seg_train eval --model outputs/seg_runs/seg_v1/best --split test

Starts from the ADE20K SegFormer-B2 encoder, replaces the decoder head with 2 classes (other, sidewalk),
trains on masks prepared by `seg_data prepare`, picks the best epoch on the VALIDATION split, and reports
sidewalk IoU for the fine-tuned model AND for the untouched pretrained model on the same photos.
The test split is only touched by `eval --split test`.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .model import select_device
from .paths import repo_path
from .preprocessing import IMAGENET_MEAN, IMAGENET_STD
from .seg_data import load_image, read_manifest
from .segment import DEFAULT_MODEL, Segmenter

ID2LABEL = {0: "other", 1: "sidewalk"}
IGNORE = 255
MIN_TRAIN, MIN_VAL = 20, 5


def load_mask(stem: str, out_dir="data/segmentation") -> np.ndarray:
    return np.array(Image.open(repo_path(out_dir) / "masks" / f"{stem}.png"))


class SegDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, crop: int = 512, long_side: int = 1024, out_dir="data/segmentation"):
        self.rows, self.train, self.crop, self.long_side, self.out_dir = rows, train, crop, long_side, out_dir
        self.jitter = T.ColorJitter(0.3, 0.3, 0.3, 0.02)
        self.mean, self.std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img, mask = load_image(repo_path(r["image"])), Image.fromarray(load_mask(r["stem"], self.out_dir))
        W, H = img.size
        scale = self.long_side / max(W, H) * (random.uniform(0.6, 1.2) if self.train else 1.0)
        size = (max(32, round(W * scale)), max(32, round(H * scale)))
        img, mask = img.resize(size, Image.BILINEAR), mask.resize(size, Image.NEAREST)
        if self.train:
            if random.random() < 0.5:
                img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.jitter(img)
        x = (T.functional.to_tensor(img) - self.mean) / self.std
        y = torch.from_numpy(np.array(mask)).long()
        if self.train:   # random crop, padding with mean colour / ignore label
            c = self.crop
            ph, pw = max(0, c - y.shape[0]), max(0, c - y.shape[1])
            if ph or pw:
                x = torch.nn.functional.pad(x, (0, pw, 0, ph), value=0.0)
                y = torch.nn.functional.pad(y, (0, pw, 0, ph), value=IGNORE)
            top, left = random.randint(0, y.shape[0] - c), random.randint(0, y.shape[1] - c)
            x, y = x[:, top:top + c, left:left + c], y[top:top + c, left:left + c]
        return x, y


def sidewalk_iou(seg: Segmenter, rows: list[dict], out_dir="data/segmentation") -> dict:
    """Pooled and mean per-image IoU of the predicted sidewalk mask vs the human mask."""
    inter = union = 0
    per = []
    for r in rows:
        gt = load_mask(r["stem"], out_dir).astype(bool)
        pred = seg.segment(load_image(repo_path(r["image"]))).sidewalk_mask
        i, u = int((pred & gt).sum()), int((pred | gt).sum())
        inter, union = inter + i, union + u
        per.append(i / u if u else 1.0)
    return {"n": len(rows), "iou_pooled": inter / union if union else None, "iou_mean": float(np.mean(per)) if per else None}


def check_enough(rows: list[dict], allow_tiny: bool) -> None:
    n = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    if not allow_tiny and (n["train"] < MIN_TRAIN or n["val"] < MIN_VAL):
        raise SystemExit(f"Refusing to train: {n} (need >= {MIN_TRAIN} train and >= {MIN_VAL} val labelled images; "
                         f"label more, or pass --allow-tiny for a plumbing test only).")


def fine_tune(rows: list[dict], out: Path, base: str = DEFAULT_MODEL, epochs: int = 30, batch_size: int = 4, lr: float = 6e-5,
              seed: int = 42, device: str = "auto", crop: int = 512, long_side: int = 1024, model=None,
              baseline: bool = True, out_dir="data/segmentation") -> dict:
    from transformers import SegformerForSemanticSegmentation
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    dev = select_device(device)
    if model is None:
        model = SegformerForSemanticSegmentation.from_pretrained(
            base, num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}, ignore_mismatched_sizes=True)
    model.to(dev)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    dl = DataLoader(SegDataset(tr, True, crop, long_side, out_dir), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=len(tr) > batch_size)
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": rest, "lr": lr}, {"params": head, "lr": lr * 10}], weight_decay=0.01)
    out.mkdir(parents=True, exist_ok=True)
    result = {"base_model": base, "seed": seed, "epochs": epochs, "n_train": len(tr), "n_val": len(va),
              "pretrained_baseline_val": None, "history": [], "best_epoch": None, "best_val": None}
    if baseline:
        try:
            result["pretrained_baseline_val"] = sidewalk_iou(Segmenter(base, device), va, out_dir)
        except OSError:
            result["pretrained_baseline_val"] = "unavailable (weights not downloadable)"
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in dl:
            loss = model(pixel_values=x.to(dev), labels=y.to(dev)).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        v = sidewalk_iou(Segmenter("in-memory", device, max_side=long_side, model=model, id2label=ID2LABEL), va, out_dir)
        result["history"].append({"epoch": ep, "train_loss": float(np.mean(losses)), **{f"val_{k}": v[k] for k in ("iou_pooled", "iou_mean")}})
        print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  val sidewalk IoU {v['iou_pooled']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if v["iou_pooled"] > best:
            best = v["iou_pooled"]
            model.save_pretrained(out / "best")
            result.update(best_epoch=ep, best_val=v)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run-name", default="seg_v1")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=6e-5)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="auto")
    t.add_argument("--allow-tiny", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True, help="fine-tuned model dir (outputs/seg_runs/<run>/best)")
    e.add_argument("--split", default="test", choices=["val", "test"])
    e.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    rows = read_manifest()
    if a.cmd == "train":
        check_enough(rows, a.allow_tiny)
        r = fine_tune(rows, repo_path("outputs/seg_runs") / a.run_name, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                      seed=a.seed, device=a.device)
        print(json.dumps({k: r[k] for k in ("pretrained_baseline_val", "best_epoch", "best_val")}, indent=2))
    else:
        sel = [r for r in rows if r["split"] == a.split]
        ft = sidewalk_iou(Segmenter(a.model, a.device), sel)
        try:
            base = sidewalk_iou(Segmenter(DEFAULT_MODEL, a.device), sel)
        except OSError:
            base = "unavailable"
        print(json.dumps({"split": a.split, "fine_tuned": ft, "pretrained_ade20k": base}, indent=2))


if __name__ == "__main__":
    main()
