"""Demo: give the trained model photos and it shows what it found.

    python -m dynow_vision.demo --images "path/to/photo_or_folder"
    python -m dynow_vision.demo --watch demo_inputs        # drop photos into the folder; each result pops up

For each photo: (1) the sidewalk finder (our fine-tuned segmenter) outlines the sidewalk, (2) the classifier
(frozen DINOv2 + our trained head, averaged over the sidewalk only) says clear / slush / snow, or declines
("uncertain", "no sidewalk found"). Results are saved as pictures in outputs/demo/results/.
Not a certification of accessibility; first results on a small dataset; no ice or snow-depth claims.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402

from .analyze_image import EXTS, make_panel  # noqa: E402
from .paths import repo_path  # noqa: E402
from .pipeline import Params, analyze_frame, build  # noqa: E402
from .preprocessing import load_rgb  # noqa: E402
from .snowheight import lookup_lines  # noqa: E402

DEFAULT_CHECKPOINT = "outputs/runs/pooled_v1/best.pt"
DEFAULT_SEG = "outputs/seg_runs/seg_v1/best"
SAY = {"clear": "CLEAR", "slush": "SLUSH / PATCHY", "snow": "SNOW", "uncertain": "UNCERTAIN (declined)",
       "no_sidewalk": "NO SIDEWALK FOUND", "unreliable_sidewalk": "SIDEWALK UNRELIABLE (declined)"}


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in EXTS and not p.name.startswith(".")) if path.is_dir() else [path]


def new_files(folder: Path, seen: set[str]) -> list[Path]:
    """Images in `folder` not yet processed (files still being copied are picked up on a later poll)."""
    out = []
    for p in list_images(folder):
        if p.name in seen:
            continue
        try:
            size1 = p.stat().st_size
            time.sleep(0.3)
            if p.stat().st_size == size1 and size1 > 0:
                out.append(p)
        except OSError:
            continue
    return out


def process(f: Path, seg, clf, out_dir: Path, params: Params) -> dict:
    img = load_rgb(f)
    r = analyze_frame(img, 0, 0.0, seg, clf, params)
    title = "sidewalk area averaged by the model" if getattr(clf, "needs_mask", False) else "crop the classifier saw"
    dest = out_dir / f"{f.stem.replace(' ', '_')}.png"
    cv2.imwrite(str(dest), make_panel(img, r, lookup_lines(f), title))
    score = f"{r.model_score:.0%}" if r.model_score is not None and r.predicted_class else ""
    return {"file": f.name, "status": r.status, "score": score, "sidewalk": f"{r.sidewalk_fraction:.0%}", "path": dest,
            "class_scores": r.class_scores}


def print_row(res: dict) -> None:
    extra = ""
    if res["class_scores"]:
        extra = "   [" + "  ".join(f"{k} {v:.0%}" for k, v in res["class_scores"].items()) + "]"
    print(f"  {res['file'][:42]:42s}  {SAY.get(res['status'], res['status']):30s} {res['score']:>5s}  sidewalk {res['sidewalk']:>4s}{extra}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", help="a photo or a folder of photos")
    g.add_argument("--watch", help="folder to watch: each photo you drop in is analysed and shown")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seg-model", default=DEFAULT_SEG)
    ap.add_argument("--out", default="outputs/demo/results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-open", action="store_true", help="do not pop up result pictures")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop watching after this long (testing)")
    a = ap.parse_args(argv)

    for p, what in ((a.checkpoint, "classifier"), (a.seg_model, "sidewalk finder")):
        if not repo_path(p).exists():
            raise SystemExit(f"{what} not found at {p}. It is created by training (see CROPS.md / SEGLABEL.md).")
    out = repo_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("loading models ...", flush=True)
    seg, clf, _ = build(str(repo_path(a.checkpoint)), a.device, str(repo_path(a.seg_model)))
    params = Params()
    print("ready.\n  file                                        result                          conf  sidewalk found\n", flush=True)

    if a.images:
        files = list_images(Path(a.images).expanduser())
        counts: dict[str, int] = {}
        results = []
        for f in files:
            res = process(f, seg, clf, out, params)
            results.append(res)
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            print_row(res)
            if not a.no_open and len(files) <= 5:
                subprocess.run(["open", str(res["path"])], check=False)
        print("\nsummary:", ", ".join(f"{SAY.get(k, k)}: {v}" for k, v in sorted(counts.items())))
        print(f"pictures saved in {out}")
        if not a.no_open and len(files) > 5:
            subprocess.run(["open", str(out)], check=False)
    else:
        folder = Path(a.watch).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        seen = {p.name for p in list_images(folder)}                 # photos already there are not re-run
        print(f"watching {folder}  (Ctrl-C to stop). Drop a photo into that folder.\n", flush=True)
        t0 = time.time()
        try:
            while a.max_seconds is None or time.time() - t0 < a.max_seconds:
                for f in new_files(folder, seen):
                    seen.add(f.name)
                    try:
                        res = process(f, seg, clf, out, params)
                    except Exception as e:                           # a bad file must not stop the demo
                        print(f"  {f.name}: could not read ({e})")
                        continue
                    print_row(res)
                    if not a.no_open:
                        subprocess.run(["open", str(res["path"])], check=False)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
