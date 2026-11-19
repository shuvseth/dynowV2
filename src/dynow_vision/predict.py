"""Classify one image and write the result to JSON.

    python -m dynow_vision.predict --image photo.jpg --checkpoint outputs/runs/<run>/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .model import load_checkpoint, select_device
from .paths import repo_path
from .preprocessing import eval_transform, load_rgb, validate_image


def predict_image(image: str | Path, checkpoint: str | Path, device: str = "auto", loaded=None) -> dict:
    ok, why = validate_image(image)
    if not ok:
        raise ValueError(f"{image}: {why}")
    dev = select_device(device)
    t_load = time.perf_counter()
    model, ck = loaded if loaded else load_checkpoint(checkpoint, dev)
    load_s = time.perf_counter() - t_load
    classes = ck["classes"]
    x = eval_transform(ck["config"]["input_size"])(load_rgb(image)).unsqueeze(0).to(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].cpu()
    if dev.type == "mps":
        torch.mps.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000
    i = int(probs.argmax())
    return {
        "input_image": str(Path(image).resolve()),
        "predicted_class": classes[i],
        "model_score": float(probs[i]),
        "class_scores": {c: float(probs[j]) for j, c in enumerate(classes)},
        "score_note": "Softmax output of the trained head. NOT a calibrated probability.",
        "model_version": f"{ck['backbone']}+head:{ck['run_name']}:{ck['checkpoint_sha256'][:12]}",
        "checkpoint": str(checkpoint),
        "device": str(dev),
        "inference_time_ms": round(infer_ms, 2),
        "model_load_time_s": round(load_s, 2),
        "scope_note": "Two-class model (clear vs snow). It cannot output 'uncertain' or detect ice, "
                      "and it classifies the whole image, not just the sidewalk.",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dynow_vision.predict", description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", help="JSON output path (default outputs/predictions/<image>.json)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    res = predict_image(a.image, a.checkpoint, a.device)
    out = Path(a.out) if a.out else repo_path("outputs/predictions") / f"{Path(a.image).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)); print(f"saved -> {out}")


if __name__ == "__main__":
    main()
