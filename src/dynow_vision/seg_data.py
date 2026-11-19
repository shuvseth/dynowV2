"""Sidewalk mask data for fine-tuning the segmenter (human-drawn labels only).

Workflow (see SEGLABEL.md):
    pip install labelme
    labelme data/raw/seg_images --autosave --nodata --labels sidewalk,no_sidewalk
    python -m dynow_vision.seg_data prepare --source "own photos Nov 2026" --license "own work" --group-regex '^(.+)_\\d+$'

Labels you draw (polygons or rectangles):
    sidewalk     the WALKABLE path surface, whatever covers it (snow, slush, ice, bare). Not the snow verge beside it.
    no_sidewalk  put a tiny shape with this label on photos that contain no sidewalk at all.
Everything outside `sidewalk` polygons becomes class "other" in the mask, so label ALL visible walkable
sidewalk in a photo. Photos with no shapes are skipped (unlabelled != no sidewalk).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .paths import repo_path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGES_DIR = "data/raw/seg_images"
ANN_DIR = "data/annotations/segmentation"      # tracked: small labelme JSON (no image data)
OUT_DIR = "data/segmentation"                  # generated masks + manifest (git-ignored)
MANIFEST_COLUMNS = ["stem", "image", "group", "split", "sidewalk_fraction", "source", "license"]


def load_image(path: str | Path) -> Image.Image:
    """RGB with EXIF rotation applied (labelme annotates the rotated image)."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def shapes_to_mask(shapes: list[dict], h: int, w: int) -> np.ndarray | None:
    """0 = other, 1 = sidewalk. None if the file has no usable shapes (unlabelled)."""
    mask = np.zeros((h, w), np.uint8)
    found = False
    for s in shapes:
        label = str(s.get("label", "")).strip().lower()
        if label == "no_sidewalk":
            found = True
        elif label == "sidewalk":
            pts = np.array(s["points"], dtype=np.float64)
            if s.get("shape_type") == "rectangle" and len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            if len(pts) < 3:
                raise ValueError(f"sidewalk shape with {len(pts)} points")
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
            found = True
    return mask if found else None


def split_groups(stems: list[str], groups: dict[str, str], seed: int, ratios=(0.7, 0.15, 0.15)) -> dict[str, str]:
    """Group-aware seeded split so near-duplicate photos never straddle train/val/test."""
    by_group: dict[str, list[str]] = {}
    for s in stems:
        by_group.setdefault(groups[s], []).append(s)
    order = sorted(by_group)
    random.Random(seed).shuffle(order)
    n = len(stems)
    targets = {"train": ratios[0] * n, "val": ratios[1] * n, "test": ratios[2] * n}
    counts = {k: 0 for k in targets}
    out: dict[str, str] = {}
    for g in order:
        # assign to the split furthest below its target (deterministic given the shuffled order)
        split = max(targets, key=lambda k: targets[k] - counts[k])
        for s in by_group[g]:
            out[s] = split
        counts[split] += len(by_group[g])
    return out


def prepare(source: str, license_: str, images_dir=IMAGES_DIR, ann_dir=ANN_DIR, out_dir=OUT_DIR, seed: int = 42,
            group_regex: str | None = None) -> dict:
    if not source.strip() or not license_.strip():
        raise ValueError("source and license are mandatory")
    images_dir, ann_dir, out_dir = repo_path(images_dir), repo_path(ann_dir), repo_path(out_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    rx = re.compile(group_regex) if group_regex else None
    rows, skipped = [], []
    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in EXTS):
        js = img_path.with_suffix(".json")
        if not js.exists():
            skipped.append((img_path.name, "no labelme json"))
            continue
        data = json.loads(js.read_text())
        img = load_image(img_path)
        w, h = img.size
        if (data.get("imageWidth"), data.get("imageHeight")) not in ((w, h), (None, None)):
            raise ValueError(f"{img_path.name}: json says {data.get('imageWidth')}x{data.get('imageHeight')} "
                             f"but image is {w}x{h} (EXIF rotation?). Re-open and re-save it in labelme.")
        mask = shapes_to_mask(data.get("shapes", []), h, w)
        if mask is None:
            skipped.append((img_path.name, "no sidewalk / no_sidewalk shapes (unlabelled)"))
            continue
        stem = img_path.stem
        Image.fromarray(mask).save(out_dir / "masks" / f"{stem}.png")
        shutil.copyfile(js, ann_dir / js.name)
        m = rx.match(stem) if rx else None
        rows.append({"stem": stem, "image": str(img_path.relative_to(repo_path("."))) if img_path.is_relative_to(repo_path(".")) else str(img_path),
                     "group": m.group(1) if m else stem, "split": "", "sidewalk_fraction": round(float(mask.mean()), 4),
                     "source": source, "license": license_})
    split = split_groups([r["stem"] for r in rows], {r["stem"]: r["group"] for r in rows}, seed)
    for r in rows:
        r["split"] = split[r["stem"]]
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, MANIFEST_COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    summary = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    return {"n_labelled": len(rows), "splits": summary, "skipped": skipped, "manifest": str(out_dir / "manifest.csv")}


def read_manifest(out_dir=OUT_DIR) -> list[dict]:
    with open(repo_path(out_dir) / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="labelme JSON -> masks + seeded group-aware split")
    p.add_argument("--source", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--images", default=IMAGES_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-regex", default=None, help="regex whose group(1) names the group, e.g. '^(.+)_\\d+$'")
    a = ap.parse_args(argv)
    r = prepare(a.source, a.license, images_dir=a.images, seed=a.seed, group_regex=a.group_regex)
    print(f"labelled images: {r['n_labelled']}  splits: {r['splits']}")
    for name, why in r["skipped"]:
        print(f"  skipped {name}: {why}")
    print(f"manifest: {r['manifest']}")


if __name__ == "__main__":
    main()
