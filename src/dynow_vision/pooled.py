"""Sidewalk condition classifier on MASK-POOLED DINOv2 patch features (clear / slush / snow).

Instead of classifying a grey-filled, squashed crop, run the frozen pretrained DINOv2 on the WHOLE photo and
average its patch features over only the sidewalk pixels. The head is a standardised multinomial logistic
regression trained by us on human-labelled crops (see CROPS.md). Dashcam road images are NOT used.

    python -m dynow_vision.pooled train --run-name pooled_v1          # fit on the train split, report val
    python -m dynow_vision.pooled eval --checkpoint outputs/runs/pooled_v1/best.pt --split test   # once, at the end
    python -m dynow_vision.pipeline --video clip.mp4 --checkpoint outputs/runs/pooled_v1/best.pt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .model import load_backbone, select_device
from .paths import repo_path
from .preprocessing import IMAGENET_MEAN, IMAGENET_STD
from .seg_data import load_image

CHECKPOINT_TYPE = "pooled"
INPUT_SIZE = 448          # multiple of 14 -> 32 x 32 patches
MIN_PATCHES = 3
MANIFEST = "data/annotations/crops_manifest.csv"
INDEX = "data/annotations/crops_index.csv"
PHOTO_DIRS = ("data/raw/seg_images", "data/raw/crop_photos")
MASK_CACHE = "data/raw/sidewalk_crops/masks"
HUMAN_MASKS = "data/segmentation/masks"
DEFAULT_C = 0.01          # chosen by group cross-validation on train+val during development


# ------------------------------------------------------------------ features
@torch.no_grad()
def extract_features(backbone, image: Image.Image, mask: np.ndarray, device, size: int = INPUT_SIZE) -> tuple[np.ndarray, int]:
    """Mean of DINOv2 patch tokens over the sidewalk. Returns (feature, number_of_patches_used)."""
    g = size // 14
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    tokens = backbone.forward_features(x)["x_norm_patchtokens"][0].float().cpu().numpy()      # (g*g, D)
    cov = cv2.resize(mask.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA).reshape(-1)
    use = cov > 0.5
    if use.sum() < MIN_PATCHES:                       # tiny sidewalk: take the best-covered patches instead
        use = cov >= np.sort(cov)[-MIN_PATCHES]
    return tokens[use].mean(0), int(use.sum())


class PooledClassifier:
    """Callable used by the pipeline: clf(crop, image=..., mask=...) -> (class, score, class_scores)."""
    needs_mask = True
    # Decline below this top-class probability. Chosen from the VALIDATION split (61% answered at 70% accuracy;
    # 0.70 answered only 27%) because this regularised model's probabilities are soft. Not calibrated.
    recommended_min_score = 0.6

    def __init__(self, backbone, payload: dict, device):
        self.backbone, self.device = backbone, device
        self.classes = list(payload["classes"])
        self.mean, self.std = payload["mean"], payload["std"]
        self.coef, self.intercept = payload["coef"], payload["intercept"]
        self.size = int(payload.get("input_size", INPUT_SIZE))

    def predict_features(self, feat: np.ndarray) -> np.ndarray:
        z = ((feat - self.mean) / self.std) @ self.coef.T + self.intercept
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def __call__(self, crop=None, image: Image.Image | None = None, mask: np.ndarray | None = None):
        if image is None or mask is None:
            raise ValueError("PooledClassifier needs the full image and the sidewalk mask")
        feat, _ = extract_features(self.backbone, image, mask, self.device, self.size)
        p = self.predict_features(feat)
        i = int(p.argmax())
        return self.classes[i], float(p[i]), {c: float(v) for c, v in zip(self.classes, p)}


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pooled_checkpoint(path) -> bool:
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("type") == CHECKPOINT_TYPE
    except Exception:
        return False


def load_pooled(path, device="auto", backbone=None) -> tuple[PooledClassifier, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("type") != CHECKPOINT_TYPE:
        raise ValueError(f"{path} is not a pooled checkpoint")
    dev = select_device(device) if isinstance(device, str) else device
    bb = backbone if backbone is not None else load_backbone(payload["backbone"]).to(dev).eval()
    payload["checkpoint_sha256"] = file_sha256(path)
    return PooledClassifier(bb, payload, dev), payload


# ------------------------------------------------------------------ training data
def _find_photo(name: str) -> Path | None:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for d in PHOTO_DIRS:
        if (repo_path(d) / name).exists():
            return repo_path(d) / name
    return None


def get_mask(photo: Path, img: Image.Image, seg=None, cache=MASK_CACHE) -> np.ndarray | None:
    """Hand-drawn mask > cached predicted mask > fresh segmenter mask (cached). None if unavailable."""
    from .pipeline import clean_mask
    human = repo_path(HUMAN_MASKS) / f"{photo.stem}.png"
    if human.exists():
        return np.array(Image.open(human)) == 1
    cached = repo_path(cache) / f"{photo.stem}.png"
    if cached.exists():
        return np.array(Image.open(cached)) == 1
    if seg is None:
        return None
    side = clean_mask(seg.segment(img).sidewalk_mask, 5)
    cached.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(side.astype(np.uint8)).save(cached)
    return side


def load_split_features(split: str, backbone, device, seg=None, classes=("clear", "slush", "snow")):
    from .dataset import read_manifest
    index = {r["crop"]: r for r in csv.DictReader(open(repo_path(INDEX)))}
    X, y, g, names, skipped = [], [], [], [], []
    for r in read_manifest(MANIFEST):
        if r["split"] != split:
            continue
        crop = r["filename"].split("/")[-1]
        photo = _find_photo(index.get(crop, {}).get("source_image", ""))
        if photo is None:
            skipped.append((crop, "source photo not found (imported video crop?)"))
            continue
        img = load_image(photo)
        mask = get_mask(photo, img, seg)
        if mask is None:
            skipped.append((crop, "no mask (pass a segmenter)"))
            continue
        feat, _ = extract_features(backbone, img, mask, device)
        X.append(feat)
        y.append(classes.index(r["label"]))
        g.append(r["group"])
        names.append(crop)
    return (np.array(X), np.array(y), np.array(g), names, skipped)


def _metrics(clf: PooledClassifier, X, y) -> dict:
    from sklearn.metrics import confusion_matrix
    if len(y) == 0:
        return {"n": 0}
    pred = np.array([int(clf.predict_features(x).argmax()) for x in X])
    cm = confusion_matrix(y, pred, labels=range(len(clf.classes)))
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()), "majority_baseline": float(np.bincount(y).max() / len(y)),
            "confusion_rows_true_cols_pred": cm.tolist(), "classes": clf.classes,
            "recall": {c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(clf.classes)}}


def train(run_name: str, C: float = DEFAULT_C, seg_model: str | None = None, device: str = "auto", out_root="outputs/runs") -> dict:
    from sklearn.linear_model import LogisticRegression
    from .segment import DEFAULT_MODEL, Segmenter
    dev = select_device(device)
    backbone = load_backbone("dinov2_vits14").to(dev).eval()
    default_seg = repo_path("outputs/seg_runs/seg_v1/best")
    seg = Segmenter(seg_model or (str(default_seg) if default_seg.exists() else DEFAULT_MODEL), device)
    classes = ("clear", "slush", "snow")
    Xtr, ytr, _, _, sk1 = load_split_features("train", backbone, dev, seg, classes)
    Xva, yva, _, _, sk2 = load_split_features("val", backbone, dev, seg, classes)
    if len(ytr) < 20 or len(set(ytr)) < len(classes):
        raise SystemExit(f"Refusing to train: {len(ytr)} train crops covering {sorted(set(ytr))} (need >= 20 and all 3 classes)")
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    lr = LogisticRegression(C=C, max_iter=4000, class_weight="balanced").fit((Xtr - mean) / std, ytr)
    payload = {"type": CHECKPOINT_TYPE, "classes": list(classes), "backbone": "dinov2_vits14", "input_size": INPUT_SIZE, "C": C,
               "mean": mean.astype(np.float32), "std": std.astype(np.float32), "coef": lr.coef_.astype(np.float32),
               "intercept": lr.intercept_.astype(np.float32), "run_name": run_name, "n_train": int(len(ytr)),
               "trained_by": "Dynow team (frozen DINOv2 + logistic head on mask-pooled patch features)",
               "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "torch_version": torch.__version__,
               "python_version": platform.python_version(), "segmenter_for_masks": str(seg.model_name)}
    out = repo_path(out_root) / run_name
    if out.exists():
        raise SystemExit(f"{out} already exists; choose another --run-name")
    out.mkdir(parents=True)
    torch.save(payload, out / "best.pt")
    clf = PooledClassifier(backbone, payload, dev)
    res = {"train": _metrics(clf, Xtr, ytr), "val": _metrics(clf, Xva, yva), "C": C,
           "skipped": sk1 + sk2, "note": "Test split untouched. Val used only to report, not to select."}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


def evaluate(checkpoint: str, split: str, seg_model: str | None = None, device: str = "auto") -> dict:
    from .segment import DEFAULT_MODEL, Segmenter
    clf, payload = load_pooled(checkpoint, device)
    default_seg = repo_path("outputs/seg_runs/seg_v1/best")
    seg = Segmenter(seg_model or (str(default_seg) if default_seg.exists() else DEFAULT_MODEL), device)
    X, y, _, _, skipped = load_split_features(split, clf.backbone, clf.device, seg, tuple(clf.classes))
    return {"split": split, **_metrics(clf, X, y), "skipped": skipped}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run-name", required=True)
    t.add_argument("--C", type=float, default=DEFAULT_C)
    t.add_argument("--seg-model", default=None)
    t.add_argument("--device", default="auto")
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--split", default="val", choices=["val", "test"])
    e.add_argument("--seg-model", default=None)
    e.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    r = train(a.run_name, a.C, a.seg_model, a.device) if a.cmd == "train" else evaluate(a.checkpoint, a.split, a.seg_model, a.device)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
