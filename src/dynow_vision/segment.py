"""Phase 2: road / sidewalk semantic segmentation with a PRETRAINED model.

The model (SegFormer fine-tuned on Cityscapes by NVIDIA) is NOT trained or fine-tuned by us.
Class ids and names are read from the model's own config, so road/sidewalk are found by
name, never by a hard-coded index.

    python -m dynow_vision.segment --input photo.jpg
    python -m dynow_vision.segment --input some_dir/ --out outputs/segmentation
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import select_device
from .paths import repo_path
from .preprocessing import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD, load_rgb, validate_image

DEFAULT_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
MODEL_NOTE = ("Pretrained SegFormer (NVIDIA) fine-tuned on Cityscapes, used as-is. Not trained by "
              "the Dynow team. Cityscapes data/weights are licensed for non-commercial research use.")

# Standard Cityscapes colours for the 19 evaluation classes (by name, so id order is irrelevant).
CITYSCAPES_COLORS = {
    "road": (128, 64, 128), "sidewalk": (244, 35, 232), "building": (70, 70, 70), "wall": (102, 102, 156),
    "fence": (190, 153, 153), "pole": (153, 153, 153), "traffic light": (250, 170, 30),
    "traffic sign": (220, 220, 0), "vegetation": (107, 142, 35), "terrain": (152, 251, 152),
    "sky": (70, 130, 180), "person": (220, 20, 60), "rider": (255, 0, 0), "car": (0, 0, 142),
    "truck": (0, 0, 70), "bus": (0, 60, 100), "train": (0, 80, 100), "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
}
LOW_CONFIDENCE = 0.5          # pixels whose top-class probability is below this are "uncertain"
MIN_SIDEWALK_FRACTION = 0.005  # below this share of the image we report "no sidewalk found"
LOW_MEAN_CONFIDENCE = 0.7


@dataclass
class SegmentationResult:
    image: Image.Image          # original RGB
    label_map: np.ndarray       # (H, W) int, model class ids
    confidence: np.ndarray      # (H, W) float, softmax prob of the chosen class
    id2label: dict[int, str]
    road_mask: np.ndarray       # (H, W) bool
    sidewalk_mask: np.ndarray   # (H, W) bool


def class_ids(id2label: dict[int, str], name: str) -> list[int]:
    ids = [i for i, n in id2label.items() if n.lower() == name]
    if not ids:
        raise KeyError(f"model has no class named {name!r}; classes: {sorted(id2label.values())}")
    return ids


def logits_to_result(logits: torch.Tensor, image: Image.Image, id2label: dict[int, str]) -> SegmentationResult:
    """logits: (1, C, h, w) at any resolution -> full-resolution masks matching `image`."""
    W, H = image.size
    probs = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False).softmax(1)[0]
    conf, label = probs.max(0)
    label = label.cpu().numpy().astype(np.int32)
    road = np.isin(label, class_ids(id2label, "road"))
    side = np.isin(label, class_ids(id2label, "sidewalk"))
    return SegmentationResult(image, label, conf.cpu().numpy(), id2label, road, side)


class Segmenter:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto", max_side: int = 1024,
                 model=None, id2label: dict[int, str] | None = None):
        self.device = select_device(device)
        self.model_name, self.max_side = model_name, max_side
        if model is None:
            from transformers import SegformerForSemanticSegmentation
            model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model = model.to(self.device).eval()
        cfg_labels = id2label if id2label is not None else model.config.id2label
        self.id2label = {int(k): v for k, v in cfg_labels.items()}

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        W, H = image.size
        s = self.max_side / max(W, H)
        w, h = max(32, round(W * s / 32) * 32), max(32, round(H * s / 32) * 32)
        arr = np.asarray(image.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def segment(self, image: Image.Image) -> SegmentationResult:
        out = self.model(pixel_values=self._prepare(image))
        logits = out.logits if hasattr(out, "logits") else out
        return logits_to_result(logits.cpu(), image, self.id2label)


# ----------------------------------------------------------------------------- outputs
def colorize(label_map: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    """Color-code by class name; unmapped / out-of-range ids are black."""
    palette = np.zeros((max(id2label) + 1, 3), dtype=np.uint8)
    for i, n in id2label.items():
        palette[i] = CITYSCAPES_COLORS.get(n.lower(), (0, 0, 0))
    return palette[np.clip(label_map, 0, len(palette) - 1)]


def make_overlay(image: Image.Image, label_map, id2label, alpha: float = 0.5) -> Image.Image:
    color = Image.fromarray(colorize(label_map, id2label))
    return Image.blend(image.convert("RGB"), color, alpha)


def sidewalk_focus(image: Image.Image, sidewalk_mask: np.ndarray) -> Image.Image:
    """Original pixels where the model says sidewalk, black elsewhere."""
    arr = np.asarray(image.convert("RGB")).copy()
    arr[~sidewalk_mask] = 0
    return Image.fromarray(arr)


def summarize(res: SegmentationResult) -> dict:
    n = res.label_map.size
    def stats(mask):
        return {"fraction_of_image": float(mask.sum() / n),
                "mean_confidence": float(res.confidence[mask].mean()) if mask.any() else None,
                "low_confidence_fraction": float((res.confidence[mask] < LOW_CONFIDENCE).mean()) if mask.any() else None}
    road, side = stats(res.road_mask), stats(res.sidewalk_mask)
    warnings = []
    if side["fraction_of_image"] < MIN_SIDEWALK_FRACTION:
        warnings.append("No (or almost no) sidewalk detected. The sidewalk may be absent, outside the frame, "
                        "or hidden/unrecognizable (e.g. under snow); the empty mask is NOT evidence either way.")
    elif side["mean_confidence"] < LOW_MEAN_CONFIDENCE or side["low_confidence_fraction"] > 0.25:
        warnings.append("Low model confidence inside the sidewalk mask; its boundary is uncertain "
                        "(common when snow covers the sidewalk). Inspect confidence.png before trusting it.")
    if road["fraction_of_image"] < MIN_SIDEWALK_FRACTION:
        warnings.append("No (or almost no) road detected.")
    return {"image_size_wh": list(res.image.size), "road": road, "sidewalk": side,
            "warnings": warnings, "confidence_note": "Softmax confidence of the pretrained model, not a calibrated probability. High confidence does NOT mean correct: snow/ice-covered sidewalks have been confidently labeled road."}


def save_outputs(res: SegmentationResult, out_dir: str | Path, extra: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.image.save(out / "original.png")
    make_overlay(res.image, res.label_map, res.id2label).save(out / "overlay.png")
    Image.fromarray(res.road_mask.astype(np.uint8) * 255).save(out / "road_mask.png")
    Image.fromarray(res.sidewalk_mask.astype(np.uint8) * 255).save(out / "sidewalk_mask.png")
    sidewalk_focus(res.image, res.sidewalk_mask).save(out / "sidewalk_only.png")
    conf8 = (np.clip(res.confidence, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(conf8).save(out / "confidence.png")
    summary = {**summarize(res), "model_note": MODEL_NOTE, **(extra or {})}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def collect_images(inp: Path) -> list[Path]:
    if inp.is_dir():
        return sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return [inp]


def process(inputs: list[Path], out_root: Path, seg: Segmenter) -> list[dict]:
    results = []
    for p in inputs:
        ok, why = validate_image(p)
        if not ok:
            print(f"skip {p}: {why}")
            results.append({"input": str(p), "error": why})
            continue
        t0 = time.perf_counter()
        res = seg.segment(load_rgb(p))
        ms = (time.perf_counter() - t0) * 1000
        s = save_outputs(res, out_root / p.stem, {"input_image": str(p.resolve()), "model": seg.model_name,
                                                  "device": str(seg.device), "inference_time_ms": round(ms, 1)})
        print(f"{p.name}: road {s['road']['fraction_of_image']:.1%}, sidewalk {s['sidewalk']['fraction_of_image']:.1%}"
              + "".join(f"\n  WARNING: {w}" for w in s["warnings"]))
        results.append({"input": str(p), **s})
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dynow_vision.segment", description=__doc__)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--out", default="outputs/segmentation", help="output root (one subfolder per image)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-side", type=int, default=1024)
    a = ap.parse_args(argv)
    inputs = collect_images(Path(a.input).expanduser())
    if not inputs:
        raise SystemExit(f"no images found in {a.input}")
    process(inputs, repo_path(a.out), Segmenter(a.model, a.device, a.max_side))


if __name__ == "__main__":
    main()
