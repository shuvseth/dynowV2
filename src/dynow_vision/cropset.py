"""Sidewalk-crop dataset: cut the sidewalk out of photos, label its CONDITION by hand, train a 3-class classifier.

    python -m dynow_vision.cropset make  --images data/raw/seg_images --source "..." --license "..." [--seg-model DIR]
    python -m dynow_vision.cropset label --annotator you
    python -m dynow_vision.cropset build
    python -m dynow_vision.train --config configs/crops3.yaml --run-name crops3_v1

Crops are exactly what the pipeline feeds the classifier (masked: everything but sidewalk is neutral grey).
Where a hand-drawn mask exists (data/segmentation/masks) it is used; otherwise the segmenter's mask is used and
the crop is marked "predicted". Labels (one keystroke each):
    c  clear    bare walking surface, no meaningful snow or ice (dry or merely wet)
    w  slush    partly covered: slush, patchy snow or ice, dirty melting snow, mixed bare/snow
    s  snow     mostly or fully covered by snow
    u  unsure   you cannot tell      x  bad crop (mask is mostly road/verge/not sidewalk)
Only c / w / s are used for training. u and x are kept in the labels file but excluded.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .dataset import COLUMNS as MANIFEST_COLUMNS, make_splits, write_manifest
from .paths import repo_path
from .preprocessing import IMAGE_EXTENSIONS
from .seg_data import load_image

CROPS_DIR = "data/raw/sidewalk_crops"
INDEX_CSV = "data/annotations/crops_index.csv"
LABELS_CSV = "data/annotations/crop_labels.csv"
MANIFEST_CSV = "data/annotations/crops_manifest.csv"
MASKS_DIR = "data/segmentation/masks"
CLASSES = ("clear", "slush", "snow")
KEYS = {"c": "clear", "w": "slush", "s": "snow", "u": "uncertain", "x": "bad_crop"}
HELP = "[c]lear  [w]slush/patchy  [s]now  [u]nsure  [x]bad crop  [Enter]=skip  [b]ack  [q]uit"
INDEX_COLUMNS = ["crop", "source_image", "group", "mask_source", "source", "license"]
LABEL_COLUMNS = ["crop", "label", "annotator", "timestamp_utc"]
PREVIEW_H = 420


# ------------------------------------------------------------------ csv helpers
def _read(path) -> list[dict]:
    p = repo_path(path)
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _write(path, cols, rows) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)


# ------------------------------------------------------------------ make
def _outline_preview(image: Image.Image, side: np.ndarray, crop: Image.Image | None) -> np.ndarray:
    """[original with sidewalk outlined | the exact crop] so the labeler sees what the mask covered."""
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    over = bgr.copy()
    over[side] = (0.55 * over[side] + 0.45 * np.array((232, 35, 244))).astype(np.uint8)
    cnts, _ = cv2.findContours(side.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (255, 255, 255), 2)
    parts = [over]
    if crop is not None:
        parts.append(cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR))
    fit = lambda im: cv2.resize(im, (max(1, int(im.shape[1] * PREVIEW_H / im.shape[0])), PREVIEW_H))
    return np.hstack([fit(p) for p in parts])


def make_crops(images: list[Path], source: str, license_: str, seg=None, out_dir=CROPS_DIR, group_regex: str | None = None,
               masks_dir=MASKS_DIR, crop_pad: float = 0.10, open_kernel: int = 5, min_pixels: int = 400) -> list[dict]:
    """One masked sidewalk crop per image. Adds rows to the index (existing entries are kept)."""
    from .pipeline import clean_mask, make_crop
    if not source.strip() or not license_.strip():
        raise ValueError("source and license are mandatory")
    out, prev = repo_path(out_dir), repo_path(out_dir) / "preview"
    out.mkdir(parents=True, exist_ok=True)
    prev.mkdir(exist_ok=True)
    rx = re.compile(group_regex) if group_regex else None
    index = {r["crop"]: r for r in _read(INDEX_CSV)}
    added = []
    for p in images:
        img = load_image(p)
        mpath = repo_path(masks_dir) / f"{p.stem}.png"
        if mpath.exists():
            m = np.array(Image.open(mpath)) == 1
            if m.shape != (img.height, img.width):
                raise ValueError(f"{p.name}: mask size {m.shape} != image {(img.height, img.width)}")
            side, msrc = m, "human"
        elif seg is not None:
            side = clean_mask(seg.segment(img).sidewalk_mask, open_kernel)
            msrc = f"predicted:{getattr(seg, 'model_name', 'segmenter')}"
        else:
            continue
        if side.sum() < min_pixels:
            continue
        crop, _ = make_crop(img, side, "masked", crop_pad)
        name = f"{p.stem}.png"
        crop.save(out / name)
        cv2.imwrite(str(prev / f"{p.stem}.jpg"), _outline_preview(img, side, crop))
        m_ = rx.match(p.stem) if rx else None
        row = {"crop": name, "source_image": p.name, "group": m_.group(1) if m_ else p.stem, "mask_source": msrc,
               "source": source, "license": license_}
        index[name] = row
        added.append(row)
    _write(INDEX_CSV, INDEX_COLUMNS, list(index.values()))
    return added


def ingest_crops(crops_dir: Path, source: str, license_: str, group: str, out_dir=CROPS_DIR) -> int:
    """Import crops the video pipeline already produced (outputs/video/<clip>/crops). All frames of one clip share a group."""
    out, prev = repo_path(out_dir), repo_path(out_dir) / "preview"
    out.mkdir(parents=True, exist_ok=True)
    prev.mkdir(exist_ok=True)
    index = {r["crop"]: r for r in _read(INDEX_CSV)}
    n = 0
    for p in sorted(q for q in crops_dir.iterdir() if q.suffix.lower() in IMAGE_EXTENSIONS):
        name = f"{group}_{p.stem}.png"
        Image.open(p).convert("RGB").save(out / name)
        shutil.copyfile(out / name, prev / f"{Path(name).stem}.png")
        index[name] = {"crop": name, "source_image": str(p), "group": group, "mask_source": "predicted:video-pipeline",
                       "source": source, "license": license_}
        n += 1
    _write(INDEX_CSV, INDEX_COLUMNS, list(index.values()))
    return n


# ------------------------------------------------------------------ label
def read_labels() -> dict[str, dict]:
    return {r["crop"]: r for r in _read(LABELS_CSV)}


def label_session(annotator: str, relabel: bool = False, input_fn=input, show_fn=None, out=print) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", annotator or ""):
        raise ValueError("annotator must be a simple name (letters, digits, _ . -)")
    index = _read(INDEX_CSV)
    labels = read_labels()
    queue = [r["crop"] for r in index if relabel or r["crop"] not in labels]
    out(HELP)
    out("  clear = bare surface (dry/wet ok) | slush = partial/patchy/dirty melt | snow = mostly covered | x = mask is wrong")
    n, pos = 0, 0
    while pos < len(queue):
        name = queue[pos]
        out(f"\n[{pos + 1}/{len(queue)}] {name}")
        if show_fn:
            show_fn(name)
        ans = input_fn("label > ").strip().lower()
        if ans == "q":
            break
        if ans == "b":
            pos = max(0, pos - 1)
            continue
        if ans == "":
            pos += 1
            continue
        if ans not in KEYS:
            out(HELP)
            continue
        labels = read_labels()
        labels[name] = {"crop": name, "label": KEYS[ans], "annotator": annotator,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _write(LABELS_CSV, LABEL_COLUMNS, list(labels.values()))
        n += 1
        pos += 1
    return n


def preview_path(name: str) -> Path:
    prev = repo_path(CROPS_DIR) / "preview"
    stem = Path(name).stem
    for ext in (".jpg", ".png"):
        if (prev / f"{stem}{ext}").exists():
            return prev / f"{stem}{ext}"
    return repo_path(CROPS_DIR) / name


def _open_in_preview(name: str) -> None:
    subprocess.run(["open", str(preview_path(name))], check=False)


# ------------------------------------------------------------------ build / status
def build_manifest(seed: int = 42) -> dict:
    """Labeled c/w/s crops -> training manifest (same columns as the Phase 1 manifest), group-aware split."""
    labels = read_labels()
    rows = []
    for r in _read(INDEX_CSV):
        lab = labels.get(r["crop"], {}).get("label", "")
        if lab in CLASSES:
            rows.append({"filename": f"sidewalk_crops/{r['crop']}", "source": f"{r['source']} [{r['mask_source']} mask]",
                         "license": r["license"], "group": r["group"], "label": lab, "split": ""})
    make_splits(rows, seed=seed, classes=CLASSES)
    write_manifest(rows, MANIFEST_CSV)
    summ = {c: {s: sum(r["label"] == c and r["split"] == s for r in rows) for s in ("train", "val", "test")} for c in CLASSES}
    return {"n": len(rows), "per_class_per_split": summ, "manifest": str(repo_path(MANIFEST_CSV))}


def status() -> dict:
    index, labels = _read(INDEX_CSV), read_labels()
    counts = {}
    for r in labels.values():
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return {"crops": len(index), "labelled": len(labels), "unlabelled": len(index) - len(labels), "label_counts": counts,
            "human_masks": sum(r["mask_source"] == "human" for r in index),
            "predicted_masks": sum(r["mask_source"] != "human" for r in index)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="cut sidewalk crops out of photos")
    m.add_argument("--images", help="folder of photos")
    m.add_argument("--from-crops", help="folder of crops from a video pipeline run (outputs/video/<clip>/crops)")
    m.add_argument("--group", help="group name for --from-crops (e.g. the clip name)")
    m.add_argument("--source", required=True)
    m.add_argument("--license", required=True)
    m.add_argument("--seg-model", default=None, help="segmenter for photos without a hand-drawn mask (default: pipeline default)")
    m.add_argument("--group-regex", default=None)
    m.add_argument("--device", default="auto")
    lb = sub.add_parser("label")
    lb.add_argument("--annotator", required=True)
    lb.add_argument("--relabel", action="store_true")
    lb.add_argument("--no-preview", action="store_true")
    sub.add_parser("build")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        if a.from_crops:
            if not a.group:
                ap.error("--from-crops needs --group")
            print(f"imported {ingest_crops(Path(a.from_crops), a.source, a.license, a.group)} crops")
        else:
            if not a.images:
                ap.error("give --images or --from-crops")
            from .segment import DEFAULT_MODEL, Segmenter
            files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            added = make_crops(files, a.source, a.license, Segmenter(a.seg_model or DEFAULT_MODEL, a.device), group_regex=a.group_regex)
            print(f"made {len(added)} crops from {len(files)} photos "
                  f"({sum(r['mask_source'] == 'human' for r in added)} with hand-drawn masks)")
    elif a.cmd == "label":
        n = label_session(a.annotator, a.relabel, show_fn=None if a.no_preview else _open_in_preview)
        print(f"saved {n} labels")
    elif a.cmd == "build":
        import json
        print(json.dumps(build_manifest(), indent=2))
    else:
        import json
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
