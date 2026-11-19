"""Manifest handling, group-aware splitting and the PyTorch dataset.

The manifest (data/annotations/manifest.csv) is the single source of truth with columns:
    filename  path relative to data/raw
    source    where the image came from (URL, dataset name, "own photo", ...)
    license   license or usage terms
    group     original video / photo-session id; ALL images of a group share one split
    label     human-assigned: clear | snow | uncertain  (empty = not yet labeled)
    split     train | val | test (assigned by `split`; empty for unlabeled/uncertain)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .paths import repo_path
from .preprocessing import IMAGE_EXTENSIONS, load_rgb, validate_image

COLUMNS = ["filename", "source", "license", "group", "label", "split"]
VALID_LABELS = ("clear", "slush", "snow", "uncertain")   # "slush" is only used by the 3-class sidewalk-crop set (cropset.py)
TRAIN_CLASSES = ("clear", "snow")  # uncertain is excluded from the binary experiment
SPLITS = ("train", "val", "test")
DEFAULT_MANIFEST = "data/annotations/manifest.csv"
DEFAULT_RAW = "data/raw"


class ManifestError(ValueError):
    pass


# --------------------------------------------------------------------------- IO
def read_manifest(path: str | Path = DEFAULT_MANIFEST) -> list[dict]:
    path = repo_path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or list(reader.fieldnames) != COLUMNS:
            raise ManifestError(f"{path}: header must be exactly {COLUMNS}, got {reader.fieldnames}")
        return [{c: (r.get(c) or "").strip() for c in COLUMNS} for r in reader]


def write_manifest(rows: list[dict], path: str | Path = DEFAULT_MANIFEST) -> None:
    """Atomic write so an interrupted labeling session cannot corrupt the CSV."""
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in COLUMNS} for r in rows)
    os.replace(tmp, path)


def manifest_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["filename"]):
        h.update("|".join(r[c] for c in COLUMNS).encode() + b"\n")
    return h.hexdigest()


# ------------------------------------------------------------------- validation
def validate_rows(rows: list[dict], raw_dir: str | Path | None = DEFAULT_RAW,
                  check_files: bool = True) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems, seen = [], set()
    raw = repo_path(raw_dir) if raw_dir is not None else None
    for i, r in enumerate(rows, start=2):  # row 1 is the header
        fn = r["filename"]
        if not fn:
            problems.append(f"line {i}: empty filename")
            continue
        if fn in seen:
            problems.append(f"line {i}: duplicate filename {fn}")
        seen.add(fn)
        if r["label"] and r["label"] not in VALID_LABELS:
            problems.append(f"line {i} ({fn}): invalid label {r['label']!r}; allowed {VALID_LABELS}")
        if r["split"] and r["split"] not in SPLITS:
            problems.append(f"line {i} ({fn}): invalid split {r['split']!r}")
        if r["label"] in TRAIN_CLASSES + ("slush",):
            for col in ("source", "license", "group"):
                if not r[col]:
                    problems.append(f"line {i} ({fn}): labeled image is missing '{col}'")
        if check_files and raw is not None:
            ok, why = validate_image(raw / fn)
            if not ok:
                problems.append(f"line {i} ({fn}): {why}")
    return problems


def check_split_integrity(rows: list[dict]) -> None:
    """Raise if any group appears in more than one split (leakage) or a split has a bad label."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["split"]:
            if r["label"] not in TRAIN_CLASSES + ("slush",):
                raise ManifestError(f"{r['filename']}: label {r['label']!r} must not have a split")
            group_splits[r["group"]].add(r["split"])
    leaked = {g: s for g, s in group_splits.items() if len(s) > 1}
    if leaked:
        raise ManifestError(f"group leakage across splits: {leaked}")


# ----------------------------------------------------------------- adding images
def add_images(inputs: list[str | Path], source: str, license: str,
               group: str | None = None, group_regex: str | None = None,
               subdir: str | None = None, manifest: str | Path = DEFAULT_MANIFEST,
               raw_dir: str | Path = DEFAULT_RAW, label_map: dict[str, str] | None = None) -> dict:
    """Copy images into data/raw and register them (unlabeled) in the manifest.

    group:        every added image shares this group (use for frames of one video).
    group_regex:  first capture group of the regex applied to the file stem is the group.
    neither:      each image is its own group.
    label_map:    for already-sorted folders, e.g. {"snow": "snow", "clear": "clear"}. The image's
                  parent folder name (case-insensitive) is used as its label; images in other
                  folders are skipped. These are third-party labels, not ours: spot-check them.
    """
    label_map = {k.lower(): v for k, v in (label_map or {}).items()}
    for v in label_map.values():
        if v not in VALID_LABELS:
            raise ManifestError(f"label_map value {v!r} not in {VALID_LABELS}")
    if not source or not license:
        raise ManifestError("source and license are required (record where every image came from)")
    rx = re.compile(group_regex) if group_regex else None
    raw = repo_path(raw_dir)
    dest_dir = raw / subdir if subdir else raw
    dest_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            files.append(p)

    rows = read_manifest(manifest)
    known = {r["filename"] for r in rows}
    result = {"added": [], "skipped_existing": [], "rejected": [], "skipped_unmapped": []}
    for f in files:
        label = ""
        if label_map:
            label = label_map.get(f.parent.name.lower(), "")
            if not label:
                result["skipped_unmapped"].append(str(f))
                continue
        ok, why = validate_image(f)
        if not ok:
            result["rejected"].append((str(f), why))
            continue
        dest = dest_dir / f.name
        rel = str(dest.relative_to(raw))
        if rel in known:
            result["skipped_existing"].append(rel)
            continue
        if dest.exists() and dest.resolve() != f.resolve():
            result["rejected"].append((str(f), f"a different file named {rel} already exists in raw/"))
            continue
        if not dest.exists():
            shutil.copy2(f, dest)
        if group:
            g = group
        elif rx:
            m = rx.search(f.stem)
            if not m or not m.groups():
                result["rejected"].append((str(f), f"group regex {group_regex!r} did not match"))
                continue
            g = m.group(1)
        else:
            g = rel
        rows.append({"filename": rel, "source": source, "license": license,
                     "group": g, "label": label, "split": ""})
        known.add(rel)
        result["added"].append(rel)
    write_manifest(rows, manifest)
    return result


# -------------------------------------------------------------------- splitting
def make_splits(rows: list[dict], ratios=(0.7, 0.15, 0.15), seed: int = 42, classes=TRAIN_CLASSES) -> list[dict]:
    """Assign train/val/test by GROUP so frames of one video never straddle splits.

    Groups are stratified by their majority label and shuffled with `seed`. Each
    group goes to the split furthest below its target share of that class's images.
    Existing splits are overwritten; uncertain/unlabeled images get no split.
    """
    if abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("ratios must sum to 1")
    for r in rows:
        r["split"] = ""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["label"] in classes:
            groups[r["group"]].append(r)
    by_class: dict[str, list[str]] = defaultdict(list)
    for g, members in groups.items():
        majority = Counter(m["label"] for m in members).most_common(1)[0][0]
        by_class[majority].append(g)

    rng = random.Random(seed)
    for cls in sorted(by_class):
        gs = sorted(by_class[cls])
        rng.shuffle(gs)
        total = sum(len(groups[g]) for g in gs)
        target = [ratio * total for ratio in ratios]
        filled = [0.0, 0.0, 0.0]
        # Seed one group into each split (when >=3 groups exist) so none is empty,
        # then give every remaining group to the split furthest below its target.
        for n, g in enumerate(gs):
            if len(gs) >= 3 and n < 3:
                idx = n
            else:
                idx = max(range(3), key=lambda i: (target[i] - filled[i]) / max(target[i], 1e-9))
            for m in groups[g]:
                m["split"] = SPLITS[idx]
            filled[idx] += len(groups[g])
    check_split_integrity(rows)
    return rows


def class_summary(rows: list[dict]) -> dict[str, dict[str, int]]:
    """{split_or_overall: {label: count}}; 'unsplit' collects labeled images without a split."""
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        out["all"][r["label"] or "unlabeled"] += 1
        if r["label"] in TRAIN_CLASSES:
            out[r["split"] or "unsplit"][r["label"]] += 1
    return {k: dict(v) for k, v in out.items()}


def format_summary(rows: list[dict]) -> str:
    s = class_summary(rows)
    lines = [f"{'':8}" + "".join(f"{c:>11}" for c in (*TRAIN_CLASSES, "uncertain", "unlabeled"))]
    for k in ("all", *SPLITS, "unsplit"):
        if k in s:
            lines.append(f"{k:8}" + "".join(f"{s[k].get(c, 0):>11}" for c in (*TRAIN_CLASSES, "uncertain", "unlabeled")))
    groups = {sp: len({r['group'] for r in rows if r['split'] == sp}) for sp in SPLITS}
    lines.append("groups per split: " + ", ".join(f"{k}={v}" for k, v in groups.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------- Dataset
class WinterDataset(Dataset):
    """Images of one split, restricted to the experiment's classes."""

    def __init__(self, rows: list[dict], split: str, transform, raw_dir=DEFAULT_RAW,
                 classes=TRAIN_CLASSES):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.classes = tuple(classes)
        self.raw = repo_path(raw_dir)
        self.transform = transform
        self.items = [r for r in rows if r["split"] == split and r["label"] in self.classes]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        path = self.raw / r["filename"]
        try:
            img = load_rgb(path)
        except Exception as e:  # surface which file is bad
            raise RuntimeError(f"failed to load {path}: {e}") from e
        return self.transform(img), self.classes.index(r["label"]), r["filename"]

    def labels(self) -> list[int]:
        return [self.classes.index(r["label"]) for r in self.items]


# -------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dynow_vision.dataset", description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="copy images into data/raw and register them")
    a.add_argument("inputs", nargs="+", help="image files or directories")
    a.add_argument("--source", required=True)
    a.add_argument("--license", required=True)
    a.add_argument("--group", help="shared group id (e.g. the source video name)")
    a.add_argument("--group-regex", help="regex; first capture group of the file stem is the group")
    a.add_argument("--subdir", help="subfolder of data/raw to copy into")
    a.add_argument("--label-map", nargs="+", metavar="FOLDER=LABEL",
                   help="take labels from parent folder names, e.g. Snow=snow Clear=clear (others skipped)")

    sub.add_parser("validate", help="check labels, metadata, files and split integrity")
    sp = sub.add_parser("split", help="assign group-aware train/val/test splits")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    sub.add_parser("summary", help="class distribution per split")

    args = ap.parse_args(argv)
    if args.cmd == "add":
        lm = dict(kv.split("=", 1) for kv in args.label_map) if args.label_map else None
        res = add_images(args.inputs, args.source, args.license, args.group, args.group_regex,
                         args.subdir, args.manifest, args.raw_dir, lm)
        print(f"added {len(res['added'])}, already registered {len(res['skipped_existing'])}, "
              f"rejected {len(res['rejected'])}, skipped (unmapped folder) {len(res['skipped_unmapped'])}")
        for f, why in res["rejected"]:
            print(f"  rejected {f}: {why}")
        return 0
    rows = read_manifest(args.manifest)
    if args.cmd == "validate":
        problems = validate_rows(rows, args.raw_dir)
        try:
            check_split_integrity(rows)
        except ManifestError as e:
            problems.append(str(e))
        print("\n".join(problems) if problems else f"OK: {len(rows)} rows valid")
        return 1 if problems else 0
    if args.cmd == "split":
        problems = validate_rows(rows, args.raw_dir)
        if problems:
            print("\n".join(problems)); return 1
        write_manifest(make_splits(rows, tuple(args.ratios), args.seed), args.manifest)
        rows = read_manifest(args.manifest)
    print(format_summary(rows) if rows else "manifest is empty or missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
