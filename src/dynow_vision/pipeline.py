"""Phase 3: video pipeline = segmentation (pretrained) -> sidewalk crop -> classifier (ours).

    python -m dynow_vision.pipeline --video clip.mp4 --checkpoint outputs/runs/udot_proxy_v1/best.pt

For each sampled frame: segment road/sidewalk, decide whether a sidewalk is reliably present,
cut a sidewalk-only crop, classify the crop as clear/snow, abstain when unsure. The classifier gives
a CROP-LEVEL label, never snow pixels. Frames between samples reuse the latest result.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .model import load_checkpoint, select_device
from .paths import repo_path
from .preprocessing import IMAGENET_MEAN, eval_transform
from .segment import DEFAULT_MODEL, MODEL_NOTE, Segmenter

# Frame outcomes. Only "clear" and "snow" are condition predictions.
NO_SIDEWALK, UNRELIABLE, UNCERTAIN, CLEAR, SNOW = "no_sidewalk", "unreliable_sidewalk", "uncertain", "clear", "snow"
STATUS_COLORS_BGR = {CLEAR: (80, 200, 80), SNOW: (255, 200, 40), UNCERTAIN: (0, 215, 255),
                     UNRELIABLE: (0, 140, 255), NO_SIDEWALK: (160, 160, 160)}
STATUS_TEXT = {CLEAR: "CLEAR", SNOW: "SNOW", UNCERTAIN: "UNCERTAIN", UNRELIABLE: "SIDEWALK UNRELIABLE",
               NO_SIDEWALK: "NO SIDEWALK FOUND"}
ROAD_BGR, SIDEWALK_BGR = (128, 64, 128)[::-1], (244, 35, 232)[::-1]   # Cityscapes colours, as BGR
FILL_RGB = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)           # neutral fill = ~0 after normalization
LIMITATIONS = [
    "The classifier was trained on dashcam ROAD images (UDOT), not sidewalks; its sidewalk behaviour is unvalidated.",
    "The segmentation model is pretrained on Cityscapes and often labels snow/ice-covered sidewalks as road with high confidence.",
    "Predictions are per-crop labels (clear/snow); they do not locate snow pixels or measure depth. Ice is not detected.",
    "Model scores are uncalibrated softmax values.",
]


@dataclass
class Params:
    sample_fps: float = 2.0
    min_sidewalk_fraction: float = 0.03   # of the frame
    min_seg_confidence: float = 0.6       # mean seg confidence inside the sidewalk mask
    min_clf_score: float = 0.7            # below this the frame is "uncertain"
    crop_mode: str = "masked"             # masked: non-sidewalk pixels neutral grey | box: plain bbox crop (includes road)
    crop_pad: float = 0.10
    open_kernel: int = 5                  # px (at segmentation resolution) for removing mask specks
    out_max_side: int = 960


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    status: str
    reason: str
    sidewalk_fraction: float
    road_fraction: float
    sidewalk_mean_seg_confidence: float | None
    predicted_class: str | None = None
    model_score: float | None = None
    class_scores: dict | None = None
    bbox_xyxy: list | None = None
    segmentation_ms: float = 0.0
    classification_ms: float = 0.0
    # kept for drawing, not serialized
    road_mask: np.ndarray | None = field(default=None, repr=False)
    sidewalk_mask: np.ndarray | None = field(default=None, repr=False)
    crop: Image.Image | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("road_mask", "sidewalk_mask", "crop")}
        return d


# ------------------------------------------------------------------ pure helpers
def clean_mask(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Remove small specks (morphological opening). Returns bool mask."""
    if kernel <= 1:
        return mask.astype(bool)
    k = np.ones((kernel, kernel), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)


def mask_bbox(mask: np.ndarray, pad: float = 0.1) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)


def make_crop(image: Image.Image, sidewalk_mask: np.ndarray, mode: str = "masked", pad: float = 0.1):
    """Sidewalk-focused crop. 'masked' hides everything that is not sidewalk so road snow cannot
    influence the sidewalk label; 'box' keeps the plain bounding box (more context, includes road)."""
    box = mask_bbox(sidewalk_mask, pad)
    if box is None:
        return None, None
    arr = np.asarray(image.convert("RGB")).copy()
    if mode == "masked":
        arr[~sidewalk_mask] = FILL_RGB
    elif mode != "box":
        raise ValueError(f"crop_mode must be 'masked' or 'box', got {mode!r}")
    x0, y0, x1, y1 = box
    return Image.fromarray(arr[y0:y1, x0:x1]), [int(x0), int(y0), int(x1), int(y1)]


def decide_status(side_frac: float, side_conf: float | None, clf_score: float | None, clf_class: str | None,
                  p: Params) -> tuple[str, str]:
    """Returns (status, reason). The classifier is only consulted when the sidewalk is reliable."""
    if side_frac < p.min_sidewalk_fraction:
        return NO_SIDEWALK, f"sidewalk covers {side_frac:.1%} of frame (< {p.min_sidewalk_fraction:.1%})"
    if side_conf is None or side_conf < p.min_seg_confidence:
        return UNRELIABLE, f"mean segmentation confidence {side_conf if side_conf is None else round(side_conf, 2)} < {p.min_seg_confidence}"
    if clf_score is None:
        return UNRELIABLE, "classifier not run"
    if clf_score < p.min_clf_score:
        return UNCERTAIN, f"classifier score {clf_score:.2f} < {p.min_clf_score}"
    return clf_class, "ok"


def classify_crop(model, ck, crop: Image.Image, device) -> tuple[str, float, dict]:
    x = eval_transform(ck["config"]["input_size"])(crop.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].cpu()
    if device.type == "mps":
        torch.mps.synchronize()
    i = int(probs.argmax())
    classes = ck["classes"]
    return classes[i], float(probs[i]), {c: float(probs[j]) for j, c in enumerate(classes)}


# ----------------------------------------------------------------- per-frame logic
def analyze_frame(image: Image.Image, frame_index: int, t: float, seg: Segmenter, clf, p: Params) -> FrameResult:
    """clf(crop) -> (class, score, class_scores)"""
    t0 = time.perf_counter()
    res = seg.segment(image)
    seg_ms = (time.perf_counter() - t0) * 1000
    side = clean_mask(res.sidewalk_mask, p.open_kernel)
    road = res.road_mask
    n = side.size
    side_frac = float(side.sum() / n)
    side_conf = float(res.confidence[side].mean()) if side.any() else None

    cls = score = scores = crop = box = None
    clf_ms = 0.0
    if side_frac >= p.min_sidewalk_fraction and side_conf is not None and side_conf >= p.min_seg_confidence:
        crop, box = make_crop(image, side, p.crop_mode, p.crop_pad)
        t1 = time.perf_counter()
        cls, score, scores = clf(crop)
        clf_ms = (time.perf_counter() - t1) * 1000
    status, reason = decide_status(side_frac, side_conf, score, cls, p)
    return FrameResult(frame_index, round(t, 3), status, reason, side_frac, float(road.sum() / n), side_conf,
                       cls if status in (CLEAR, SNOW, UNCERTAIN) else None, score, scores, box,
                       round(seg_ms, 1), round(clf_ms, 1), road, side, crop)


# ----------------------------------------------------------------------- drawing
def annotate(frame_bgr: np.ndarray, r: FrameResult, held: bool, max_side: int) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    s = max_side / max(H, W)
    if s < 1:
        frame_bgr = cv2.resize(frame_bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.astype(np.float32)
    road = cv2.resize(r.road_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    side = cv2.resize(r.sidewalk_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    for m, c, a in ((road, ROAD_BGR, 0.35), (side, SIDEWALK_BGR, 0.45)):
        out[m] = out[m] * (1 - a) + np.array(c, np.float32) * a
    out = out.astype(np.uint8)
    color = STATUS_COLORS_BGR[r.status]
    cnts, _ = cv2.findContours(side.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    label = STATUS_TEXT[r.status] + (f"  {r.model_score:.2f}" if r.model_score is not None and r.status in (CLEAR, SNOW, UNCERTAIN) else "")
    scale = max(0.5, w / 900)
    th = max(1, round(scale * 2))
    bar_h = round(70 * scale)
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(out, label, (10, round(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.9, color, th, cv2.LINE_AA)
    sub = f"t={r.timestamp_s:.1f}s  sidewalk {r.sidewalk_fraction:.0%}  road {r.road_fraction:.0%}" + (" (held)" if held else "")
    cv2.putText(out, sub, (10, round(58 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.55, (255, 255, 255), max(1, th - 1), cv2.LINE_AA)
    note = "crop-level label, not snow pixels | uncalibrated score"
    cv2.putText(out, note, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------- the pipeline
def run_video(video: str | Path, out_dir: str | Path, seg: Segmenter, clf, p: Params,
              model_info: dict, max_frames: int | None = None, save_masks: bool = True) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / p.sample_fps))
    out = Path(out_dir)
    for sub in ("masks", "crops", "representative"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    s = p.out_max_side / max(H, W)
    ow, oh = (round(W * s), round(H * s)) if s < 1 else (W, H)
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, (ow, oh))
    if not writer.isOpened():
        raise RuntimeError("could not open video writer (avc1)")

    results: list[FrameResult] = []
    best: dict[str, tuple[float, np.ndarray]] = {}
    last: FrameResult | None = None
    t_start, idx = time.perf_counter(), 0
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and idx >= max_frames):
            break
        held = idx % step != 0
        if not held:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last = analyze_frame(img, idx, idx / fps, seg, clf, p)
            results.append(last)
            if save_masks:
                Image.fromarray(last.road_mask.astype(np.uint8) * 255).save(out / "masks" / f"frame{idx:06d}_road.png")
                Image.fromarray(last.sidewalk_mask.astype(np.uint8) * 255).save(out / "masks" / f"frame{idx:06d}_sidewalk.png")
            if last.crop is not None:
                last.crop.save(out / "crops" / f"frame{idx:06d}.jpg", quality=92)
        ann = annotate(frame, last, held, p.out_max_side)
        writer.write(ann)
        if not held:  # representative = most confident frame per status (first frame for non-condition statuses)
            key = last.status
            val = last.model_score or 0.0
            if key not in best or val > best[key][0]:
                best[key] = (val, ann)
        idx += 1
    cap.release(); writer.release()
    wall = time.perf_counter() - t_start
    for status, (_, ann) in best.items():
        cv2.imwrite(str(out / "representative" / f"{status}.jpg"), ann)

    counts = {k: sum(r.status == k for r in results) for k in (CLEAR, SNOW, UNCERTAIN, UNRELIABLE, NO_SIDEWALK)}
    n = max(len(results), 1)
    report = {
        "video": {"path": str(Path(video).resolve()), "width": W, "height": H, "fps": fps,
                  "frames_total": total, "frames_read": idx, "duration_s": round(idx / fps, 2)},
        "run": {"started_utc": started_at, "wall_time_s": round(wall, 2),
                "sampled_frames": len(results), "sample_step_frames": step,
                "processing_speed_x_realtime": round((idx / fps) / wall, 3) if wall > 0 else None,
                "mean_segmentation_ms_per_sampled_frame": round(sum(r.segmentation_ms for r in results) / n, 1),
                "mean_classification_ms_per_classified_frame": round(
                    sum(r.classification_ms for r in results if r.classification_ms) /
                    max(sum(1 for r in results if r.classification_ms), 1), 1)},
        "models": model_info,
        "params": p.__dict__,
        "summary": {"counts": counts, "fractions": {k: v / n for k, v in counts.items()}},
        "limitations": LIMITATIONS,
        "frames": [r.to_json() for r in results],
        "outputs": {"annotated_video": str(out / "annotated.mp4")},
    }
    (out / "predictions.json").write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------ comparison with human labels
def compare_with_labels(predictions_json: str | Path, labels_csv: str | Path) -> dict:
    """labels_csv columns: frame_index,label  (label in clear|snow|uncertain). Reports honest coverage."""
    rep = json.loads(Path(predictions_json).read_text())
    pred = {f["frame_index"]: f["status"] for f in rep["frames"]}
    rows = list(csv.DictReader(open(labels_csv)))
    labeled = [(int(r["frame_index"]), r["label"].strip().lower()) for r in rows if r["label"].strip().lower() in ("clear", "snow")]
    missing = [i for i, _ in labeled if i not in pred]
    labeled = [(i, l) for i, l in labeled if i in pred]
    answered = [(i, l) for i, l in labeled if pred[i] in (CLEAR, SNOW)]
    correct = sum(pred[i] == l for i, l in answered)
    return {"n_labeled_clear_or_snow": len(labeled), "n_not_in_sampled_frames": len(missing),
            "n_answered_by_pipeline": len(answered),
            "coverage": len(answered) / len(labeled) if labeled else None,
            "accuracy_on_answered": correct / len(answered) if answered else None,
            "accuracy_counting_abstentions_as_wrong": correct / len(labeled) if labeled else None,
            "confusion": {f"true_{l}_pred_{pred[i]}": sum(1 for j, m in labeled if m == l and pred[j] == pred[i]) for i, l in labeled},
            "note": "Tiny human-labeled sets give wide uncertainty; do not present as general accuracy."}


def build(checkpoint, device="auto", seg_model=DEFAULT_MODEL):
    dev = select_device(device)
    model, ck = load_checkpoint(checkpoint, dev)
    seg = Segmenter(seg_model, device)
    info = {
        "classifier": {"model_version": f"{ck['backbone']}+head:{ck['run_name']}:{ck['checkpoint_sha256'][:12]}",
                       "checkpoint": str(checkpoint), "trained_by": "Dynow team (frozen DINOv2 + trained head)",
                       "classes": ck["classes"], "device": str(dev)},
        "segmenter": {"model": seg_model, "device": str(seg.device), "trained_by": "NVIDIA (pretrained, unmodified)",
                      "note": MODEL_NOTE},
    }
    return seg, (lambda crop: classify_crop(model, ck, crop, dev)), info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dynow_vision.pipeline", description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", help="output dir (default outputs/video/<video name>)")
    ap.add_argument("--sample-fps", type=float, default=Params.sample_fps)
    ap.add_argument("--max-frames", type=int, help="stop after this many video frames (for quick tests)")
    ap.add_argument("--min-sidewalk-fraction", type=float, default=Params.min_sidewalk_fraction)
    ap.add_argument("--min-seg-confidence", type=float, default=Params.min_seg_confidence)
    ap.add_argument("--min-clf-score", type=float, default=Params.min_clf_score)
    ap.add_argument("--crop-mode", choices=["masked", "box"], default=Params.crop_mode)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--labels", help="optional CSV (frame_index,label) of human labels to compare against")
    a = ap.parse_args(argv)
    p = Params(a.sample_fps, a.min_sidewalk_fraction, a.min_seg_confidence, a.min_clf_score, a.crop_mode)
    out = repo_path(a.out) if a.out else repo_path("outputs/video") / Path(a.video).stem
    seg, clf, info = build(a.checkpoint, a.device)
    rep = run_video(Path(a.video).expanduser(), out, seg, clf, p, info, a.max_frames)
    print(json.dumps({"summary": rep["summary"]["counts"], "run": rep["run"], "video": rep["outputs"]}, indent=2))
    if a.labels:
        cmp = compare_with_labels(out / "predictions.json", a.labels)
        (out / "label_comparison.json").write_text(json.dumps(cmp, indent=2))
        print(json.dumps(cmp, indent=2))


if __name__ == "__main__":
    main()
