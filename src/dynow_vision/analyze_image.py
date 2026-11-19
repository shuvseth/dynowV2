"""Single-image version of the Phase 3 pipeline, with a picture of what it did.

    python -m dynow_vision.analyze_image --image photo.jpg --checkpoint outputs/runs/udot_plus10_v1/best.pt
    python -m dynow_vision.analyze_image --image some_folder/ --checkpoint ... --open

Writes outputs/analysis/<name>.png: [original | segmentation overlay | exact crop the classifier saw]
with the pipeline status, class and score. Same code path as the video pipeline (pretrained segmenter,
our classifier, abstention rules).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .segment import DEFAULT_MODEL
from .pipeline import (ROAD_BGR, SIDEWALK_BGR, STATUS_COLORS_BGR, STATUS_TEXT, Params, analyze_frame, build)
from .paths import repo_path
from .preprocessing import load_rgb
from .snowheight import lookup_lines

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
H = 480


def _fit(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (max(1, int(w * H / h)), H))


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_panel(image: Image.Image, r, extra: list[str] | None = None, crop_title: str = "crop the classifier saw") -> np.ndarray:
    orig = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    over = orig.copy()
    for mask, col in ((r.road_mask, ROAD_BGR), (r.sidewalk_mask, SIDEWALK_BGR)):
        m = mask.astype(bool)
        if m.shape != over.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (over.shape[1], over.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        over[m] = (0.5 * over[m] + 0.5 * np.array(col)).astype(np.uint8)
    if r.bbox_xyxy:
        x0, y0, x1, y1 = [int(v) for v in r.bbox_xyxy]
        cv2.rectangle(over, (x0, y0), (x1, y1), (255, 255, 255), 2)
    if r.crop is not None:
        crop = cv2.cvtColor(np.array(r.crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        crop = np.full((H, H, 3), 40, np.uint8)
        cv2.putText(crop, "no crop: classifier not run", (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    parts = [_label(_fit(orig), "original"),
             _label(_fit(over), "road=purple sidewalk=pink box=crop"),
             _label(_fit(crop), crop_title)]
    panel = np.hstack(parts)
    score = f" {r.predicted_class} {r.model_score:.2f}" if r.model_score is not None else ""
    extra = extra or []
    bh = 54 + 20 * len(extra)
    banner = np.full((bh, panel.shape[1], 3), 25, np.uint8)
    cv2.rectangle(banner, (0, 0), (14, bh), STATUS_COLORS_BGR[r.status], -1)
    for k, line in enumerate(extra):
        cv2.putText(banner, line, (24, 64 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, f"{STATUS_TEXT[r.status]}{score}", (24, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, f"{r.reason} | sidewalk {r.sidewalk_fraction:.1%} of image", (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([banner, panel])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(repo_path("outputs/analysis")))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seg-model", default=DEFAULT_MODEL, help="Hugging Face SegFormer id (must have road + sidewalk classes)")
    ap.add_argument("--open", action="store_true", help="open each result in Preview")
    a = ap.parse_args(argv)

    src = Path(a.image)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seg, clf, _ = build(a.checkpoint, a.device, a.seg_model)
    p = Params()
    for f in files:
        img = load_rgb(f)
        r = analyze_frame(img, 0, 0.0, seg, clf, p)
        dest = out / f"{f.stem.replace(' ', '_')}.png"
        cv2.imwrite(str(dest), make_panel(img, r, lookup_lines(f), "sidewalk area averaged by the model" if getattr(clf, "needs_mask", False) else "crop the classifier saw"))
        print(f"{f.name}: {r.status}" + (f" ({r.predicted_class} {r.model_score:.2f})" if r.model_score is not None else "")
              + f" -> {dest}")
        if a.open:
            subprocess.run(["open", str(dest)])


if __name__ == "__main__":
    main()
