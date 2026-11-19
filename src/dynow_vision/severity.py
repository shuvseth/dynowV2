"""Phase 4: preliminary, HUMAN-ASSIGNED obstruction severity (1-5 or not assessable).

There is NO automated severity model in this repo. Severity values come only from human annotations,
stored per annotator in data/annotations/severity/<annotator>.csv, separate from any model output.
The rubric is a research instrument, not a validated accessibility certification. See SEVERITY.md.

    python -m dynow_vision.severity frames --video clip.mp4          # export frames to annotate
    python -m dynow_vision.severity annotate --annotator alice --images data/raw/severity_frames/clip
    python -m dynow_vision.severity agreement                        # inter-annotator agreement
    python -m dynow_vision.severity status                           # data counts, model eligibility
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2

from .paths import repo_path
from .preprocessing import IMAGE_EXTENSIONS, validate_image

RUBRIC_VERSION = "0.1-preliminary"
UNKNOWN = "unknown"
LEVELS = {
    1: "Clear: walking surface visible, continuous, no meaningful snow cover",
    2: "Minor: patches/light snow on part of the surface; continuous passage obvious",
    3: "Moderate: substantial or uneven/packed snow; a continuous passage is still visibly present",
    4: "Severe: surface mostly covered with no clearly defined cleared passage, OR snowbank partly blocks the path/curb ramp",
    5: "Blocked: no continuous passage visible (e.g. snowbank across the full path, ramp buried)",
}
UNKNOWN_DESC = "Not assessable: sidewalk not visible/occluded/too dark/no sidewalk in view, or you cannot tell"
COLUMNS = ["image_id", "filename", "source_video", "frame_index", "annotator", "severity",
           "snowbank_across_path", "curb_ramp_blocked", "notes", "timestamp_utc"]
DEFAULT_DIR = "data/annotations/severity"
FRAME_NAME = re.compile(r"^(?P<stem>.+)_f(?P<idx>\d+)\.(jpg|jpeg|png)$", re.I)
DISCLAIMER = ("Preliminary research rubric, human-assigned. Not a validated accessibility assessment or "
              "certification; not a physical snow-depth measurement; transparent ice is not reliably visible in RGB.")
# Judgment-call thresholds before ANY automated severity model would be worth training.
MIN_IMAGES, MIN_PER_LEVEL, MIN_KAPPA = 150, 15, 0.6


# ----------------------------------------------------------------------- IO
def image_id(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def parse_frame_name(name: str) -> tuple[str, str]:
    m = FRAME_NAME.match(Path(name).name)
    return (m["stem"], str(int(m["idx"]))) if m else ("", "")


def annotator_file(annotator: str, directory: str | Path = DEFAULT_DIR) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", annotator or ""):
        raise ValueError("annotator must be a simple name (letters, digits, _ . -)")
    return repo_path(directory) / f"{annotator}.csv"


def parse_severity(value) -> int | str:
    v = str(value).strip().lower()
    if v in ("u", UNKNOWN, "unknown", "na"):
        return UNKNOWN
    if v in {"1", "2", "3", "4", "5"}:
        return int(v)
    raise ValueError(f"severity must be 1-5 or 'u' (not assessable), got {value!r}")


def read_annotations(directory: str | Path = DEFAULT_DIR) -> list[dict]:
    rows = []
    d = repo_path(directory)
    for f in sorted(d.glob("*.csv")) if d.exists() else []:
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            if list(reader.fieldnames or []) != COLUMNS:
                raise ValueError(f"{f}: header must be {COLUMNS}")
            for r in reader:
                r["severity"] = parse_severity(r["severity"])
                rows.append(r)
    return rows


def write_annotator_rows(annotator: str, rows: list[dict], directory: str | Path = DEFAULT_DIR) -> None:
    path = annotator_file(annotator, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in COLUMNS} for r in rows)
    os.replace(tmp, path)


def make_row(path: Path, annotator: str, severity, snowbank="not_recorded", ramp="not_recorded", notes="") -> dict:
    stem, idx = parse_frame_name(path.name)
    return {"image_id": image_id(path), "filename": path.name, "source_video": stem, "frame_index": idx,
            "annotator": annotator, "severity": parse_severity(severity), "snowbank_across_path": snowbank,
            "curb_ramp_blocked": ramp, "notes": notes.replace("\n", " "),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# ----------------------------------------------------------------- annotation
PROMPT_HELP = "severity [1-5, u=not assessable, Enter=skip, b=back, q=quit] > "


def annotate_session(images: list[Path], annotator: str, directory=DEFAULT_DIR, redo=False,
                     input_fn=input, show_fn=None, out=print) -> int:
    """Interactive loop; saved after every answer. Returns number of annotations written."""
    existing = {r["image_id"]: r for r in read_annotator(annotator, directory)}
    queue = [p for p in images if redo or image_id(p) not in existing]
    out("Rubric (" + RUBRIC_VERSION + "):")
    for k, v in LEVELS.items():
        out(f"  {k}: {v}")
    out(f"  u: {UNKNOWN_DESC}")
    n, pos = 0, 0
    while pos < len(queue):
        p = queue[pos]
        out(f"\n[{pos + 1}/{len(queue)}] {p.name}")
        if show_fn:
            show_fn(p)
        ans = input_fn(PROMPT_HELP).strip().lower()
        if ans == "q":
            break
        if ans == "b":
            pos = max(0, pos - 1)
            continue
        if ans == "":
            pos += 1
            continue
        try:
            sev = parse_severity(ans)
        except ValueError as e:
            out(str(e))
            continue
        snow, ramp, notes = "not_recorded", "not_recorded", ""
        if sev != UNKNOWN:
            fl = input_fn("flags: s=snowbank across path, r=curb ramp blocked, n=neither, Enter=not recorded > ").strip().lower()
            if "n" in fl:
                snow, ramp = "no", "no"
            if "s" in fl:
                snow = "yes"
            if "r" in fl:
                ramp = "yes"
        notes = input_fn("notes (optional) > ").strip()
        rows = {r["image_id"]: r for r in read_annotator(annotator, directory)}
        row = make_row(p, annotator, sev, snow, ramp, notes)
        rows[row["image_id"]] = row
        write_annotator_rows(annotator, list(rows.values()), directory)
        n += 1
        pos += 1
    return n


def read_annotator(annotator: str, directory=DEFAULT_DIR) -> list[dict]:
    f = annotator_file(annotator, directory)
    if not f.exists():
        return []
    return [r for r in read_annotations(directory) if r["annotator"] == annotator]


# ----------------------------------------------------------- consensus / stats
def consensus(values: list) -> dict:
    """Combine several annotators' severities for one image. Disagreement is reported, never averaged away."""
    n = len(values)
    nums = [v for v in values if v != UNKNOWN]
    unk = n - len(nums)
    if n == 0:
        return {"status": "not_annotated", "value": None, "range": None, "n_annotators": 0, "display": "not annotated"}
    base = {"n_annotators": n}
    if unk == n:
        return {**base, "status": "unknown", "value": None, "range": None, "display": "NOT ASSESSABLE"}
    if unk:
        return {**base, "status": "disputed", "value": None, "range": [min(nums), max(nums)],
                "display": "DISPUTED (assessable vs not assessable)"}
    lo, hi = min(nums), max(nums)
    if n == 1:
        return {**base, "status": "single", "value": nums[0], "range": [lo, hi], "display": f"{nums[0]}/5 (1 annotator)"}
    if lo == hi:
        return {**base, "status": "agreed", "value": lo, "range": [lo, hi], "display": f"{lo}/5 ({n} annotators agree)"}
    if hi - lo == 1:
        return {**base, "status": "close", "value": None, "range": [lo, hi], "display": f"{lo}-{hi}/5 (annotators differ by 1)"}
    return {**base, "status": "disputed", "value": None, "range": [lo, hi], "display": f"DISPUTED ({lo}-{hi})"}


def consensus_by_frame(annotations: list[dict], video_stem: str) -> dict[int, dict]:
    by_frame: dict[int, list] = defaultdict(list)
    for r in annotations:
        if r["source_video"] == video_stem and r["frame_index"] != "":
            by_frame[int(r["frame_index"])].append(r["severity"])
    return {i: consensus(v) for i, v in by_frame.items()}


def agreement(annotations: list[dict]) -> dict:
    """Pairwise inter-annotator agreement on images labeled by both annotators."""
    by_ann: dict[str, dict[str, object]] = defaultdict(dict)
    for r in annotations:
        by_ann[r["annotator"]][r["image_id"]] = r["severity"]
    names = sorted(by_ann)
    out = {"annotators": names, "pairs": [], "notes": []}
    if len(names) < 2:
        out["notes"].append("Need at least 2 annotators labeling the same images to measure agreement.")
        return out
    from sklearn.metrics import cohen_kappa_score
    for a, b in combinations(names, 2):
        shared = sorted(set(by_ann[a]) & set(by_ann[b]))
        pair = {"annotators": [a, b], "n_shared": len(shared)}
        if not shared:
            pair["note"] = "no images in common"
            out["pairs"].append(pair)
            continue
        va, vb = [by_ann[a][i] for i in shared], [by_ann[b][i] for i in shared]
        both = [(x, y) for x, y in zip(va, vb) if x != UNKNOWN and y != UNKNOWN]
        pair.update({
            "exact_agreement_all": sum(x == y for x, y in zip(va, vb)) / len(shared),
            "assessable_disagreement": sum((x == UNKNOWN) != (y == UNKNOWN) for x, y in zip(va, vb)),
            "n_both_assessable": len(both),
            "within_one_level": sum(abs(x - y) <= 1 for x, y in both) / len(both) if both else None,
        })
        if both and len({x for x, _ in both} | {y for _, y in both}) > 1:
            pair["quadratic_weighted_kappa"] = float(cohen_kappa_score([x for x, _ in both], [y for _, y in both],
                                                                        weights="quadratic", labels=[1, 2, 3, 4, 5]))
        else:
            pair["quadratic_weighted_kappa"] = None
            pair["note"] = "kappa undefined (all ratings identical or none assessable)"
        if len(shared) < 30:
            pair["warning"] = f"only {len(shared)} shared images; agreement estimates are very uncertain"
        out["pairs"].append(pair)
    return out


def status(annotations: list[dict]) -> dict:
    ids = defaultdict(list)
    for r in annotations:
        ids[r["image_id"]].append(r["severity"])
    cons = {i: consensus(v) for i, v in ids.items()}
    level_counts = Counter(c["value"] for c in cons.values() if c["value"] is not None)
    multi = sum(1 for v in ids.values() if len(v) >= 2)
    ag = agreement(annotations)
    kappas = [p["quadratic_weighted_kappa"] for p in ag["pairs"] if p.get("quadratic_weighted_kappa") is not None]
    reasons = []
    if sum(level_counts.values()) < MIN_IMAGES:
        reasons.append(f"{sum(level_counts.values())} images with a single agreed level (< {MIN_IMAGES})")
    short = [k for k in range(1, 6) if level_counts.get(k, 0) < MIN_PER_LEVEL]
    if short:
        reasons.append(f"levels with < {MIN_PER_LEVEL} images: {short}")
    if not kappas or min(kappas) < MIN_KAPPA:
        reasons.append(f"inter-annotator quadratic kappa not demonstrated >= {MIN_KAPPA}")
    return {"n_images_annotated": len(ids), "n_annotators": len({r['annotator'] for r in annotations}),
            "n_images_with_2plus_annotators": multi, "agreed_level_counts": dict(sorted(level_counts.items())),
            "not_assessable_images": sum(c["status"] == "unknown" for c in cons.values()),
            "disputed_images": sum(c["status"] == "disputed" for c in cons.values()),
            "model_training_eligible": not reasons, "blocking_reasons": reasons,
            "note": "No automated severity model exists. Severity is human-assigned only."}


# ------------------------------------------------------------- frame export
def export_frames(video: str | Path, out_dir: str | Path, sample_fps: float = 2.0, max_frames: int | None = None) -> list[Path]:
    """Save frames at the SAME indices the video pipeline samples, named <stem>_f<index>.jpg."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    stem, out = Path(video).stem, Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and idx >= max_frames):
            break
        if idx % step == 0:
            p = out / f"{stem}_f{idx:06d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved.append(p)
        idx += 1
    cap.release()
    return saved


def _open_in_preview(path) -> None:
    subprocess.run(["open", str(path)], check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dynow_vision.severity", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="annotation directory (one CSV per annotator)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames", help="export frames from a video for annotation")
    f.add_argument("--video", required=True); f.add_argument("--out"); f.add_argument("--sample-fps", type=float, default=2.0)
    f.add_argument("--max-frames", type=int)
    a = sub.add_parser("annotate", help="assign severity to images")
    a.add_argument("--annotator", required=True); a.add_argument("--images", required=True)
    a.add_argument("--redo", action="store_true"); a.add_argument("--no-open", action="store_true")
    sub.add_parser("agreement", help="inter-annotator agreement")
    sub.add_parser("status", help="annotation counts and model-training eligibility")
    args = ap.parse_args(argv)

    import json
    if args.cmd == "frames":
        out = Path(args.out) if args.out else repo_path("data/raw/severity_frames") / Path(args.video).stem
        saved = export_frames(Path(args.video).expanduser(), out, args.sample_fps, args.max_frames)
        print(f"saved {len(saved)} frames to {out}")
    elif args.cmd == "annotate":
        d = Path(args.images).expanduser()
        imgs = sorted(p for p in (d.rglob("*") if d.is_dir() else [d]) if p.suffix.lower() in IMAGE_EXTENSIONS and validate_image(p)[0])
        if not imgs:
            raise SystemExit(f"no valid images in {d}")
        n = annotate_session(imgs, args.annotator, args.dir, args.redo, show_fn=None if args.no_open else _open_in_preview)
        print(f"\n{n} annotation(s) saved to {annotator_file(args.annotator, args.dir)}")
    elif args.cmd == "agreement":
        print(json.dumps(agreement(read_annotations(args.dir)), indent=2))
    else:
        print(json.dumps(status(read_annotations(args.dir)), indent=2))


if __name__ == "__main__":
    main()
