"""Demo pictures from REAL results (nothing hand-edited): metrics chart, segmenter training curve, prediction sheets.

    python -m dynow_vision.demo_assets --checkpoint outputs/runs/pooled_v1/best.pt --split test --out outputs/demo

Prediction sheets show every photo of the chosen split (not a hand-picked subset): green border = model matched the
human label, red = it did not. True labels are the human labels from CROPS.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .dataset import read_manifest  # noqa: E402
from .paths import repo_path  # noqa: E402
from .pooled import INDEX, MANIFEST, _find_photo, _metrics, extract_features, get_mask, load_pooled  # noqa: E402
from .seg_data import load_image  # noqa: E402

COLORS = {"clear": "#2e9e5b", "slush": "#e08a1e", "snow": "#3a7bd5"}


def predict_split(checkpoint, split):
    clf, payload = load_pooled(checkpoint, "auto")
    index = {r["crop"]: r for r in csv.DictReader(open(repo_path(INDEX)))}
    out, X, y = [], [], []
    for r in read_manifest(MANIFEST):
        if r["split"] != split:
            continue
        crop = r["filename"].split("/")[-1]
        photo = _find_photo(index[crop]["source_image"])
        img = load_image(photo)
        mask = get_mask(photo, img)
        feat, _ = extract_features(clf.backbone, img, mask, clf.device)
        p = clf.predict_features(feat)
        X.append(feat)
        y.append(clf.classes.index(r["label"]))
        out.append({"name": photo.name, "img": img, "mask": mask, "true": r["label"], "pred": clf.classes[int(p.argmax())], "p": float(p.max())})
    return clf, payload, out, _metrics(clf, np.array(X), np.array(y))


def metrics_figure(m_val, m_test, classes, path, n_train):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    cm = np.array(m_test["confusion_rows_true_cols_pred"])
    ax[0].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=15)
    ax[0].set_xticks(range(3), classes); ax[0].set_yticks(range(3), classes)
    ax[0].set_xlabel("model said"); ax[0].set_ylabel("human label"); ax[0].set_title(f"Held-out test photos (n={m_test['n']})")
    bars = {"always guess\nthe biggest class": m_test["majority_baseline"], f"validation\n(n={m_val['n']})": m_val["accuracy"], f"test\n(n={m_test['n']})": m_test["accuracy"]}
    b = ax[1].bar(list(bars), list(bars.values()), color=["#9aa0a6", "#7fb0e8", "#2e6fd0"])
    ax[1].set_ylim(0, 1); ax[1].set_title("Accuracy"); ax[1].bar_label(b, fmt="%.0f%%", labels=[f"{v * 100:.0f}%" for v in bars.values()])
    rec = m_test["recall"]
    b2 = ax[2].bar(list(rec), [rec[c] or 0 for c in rec], color=[COLORS[c] for c in rec])
    ax[2].set_ylim(0, 1); ax[2].set_title("Test: share of each class found"); ax[2].bar_label(b2, labels=[f"{(rec[c] or 0) * 100:.0f}%" for c in rec])
    fig.suptitle(f"Sidewalk condition classifier (trained on {n_train} human-labelled crops)  -  small test set, first results", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def segmenter_figure(path):
    mp = repo_path("outputs/seg_runs/seg_v1/metrics.json")
    if not mp.exists():
        return False
    r = json.loads(mp.read_text())
    h = r["history"]; ep = [x["epoch"] for x in h]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [x["train_loss"] for x in h], color="#c0392b"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_title("Sidewalk finder: training loss")
    ax[1].plot(ep, [x["val_iou_pooled"] for x in h], color="#2e6fd0", label="fine-tuned (ours)")
    base = r.get("pretrained_baseline_val")
    if isinstance(base, dict):
        ax[1].axhline(base["iou_pooled"], color="#9aa0a6", ls="--", label="pretrained, no training")
    ax[1].axvline(r["best_epoch"], color="#2e9e5b", ls=":", label=f"saved model (epoch {r['best_epoch']})")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("sidewalk overlap with human outline (IoU)"); ax[1].set_title("Validation, 6 photos"); ax[1].legend(); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def tile(item, w=420):
    img = np.array(item["img"].convert("RGB")).copy()
    m = item["mask"]
    if m.shape != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    img[m] = (0.55 * img[m] + 0.45 * np.array((232, 35, 244))).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (255, 255, 255), 2)
    h = int(img.shape[0] * w / img.shape[1])
    img = cv2.resize(img, (w, h))[:300]
    img = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    ok = item["true"] == item["pred"]
    canvas = np.full((300 + 52, w, 3), 255, np.uint8)
    canvas[:img.shape[0]] = img[:300]
    cv2.rectangle(canvas, (0, 300), (w, 352), (60, 160, 70) if ok else (200, 60, 60), -1)
    cv2.putText(canvas, f"human: {item['true']}", (8, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"model: {item['pred']} ({item['p']:.0%})", (8, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def prediction_sheets(items, out_dir, per_sheet=12, cols=4):
    paths = []
    for s in range(0, len(items), per_sheet):
        tiles = [tile(t) for t in items[s:s + per_sheet]]
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        p = out_dir / f"predictions_{s // per_sheet + 1}.png"
        cv2.imwrite(str(p), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="outputs/demo")
    a = ap.parse_args(argv)
    out = repo_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clf, payload, items, m_test = predict_split(a.checkpoint, a.split)
    _, _, _, m_val = predict_split(a.checkpoint, "val")
    metrics_figure(m_val, m_test, clf.classes, out / "metrics.png", payload["n_train"])
    seg_ok = segmenter_figure(out / "segmenter_training.png")
    sheets = prediction_sheets(items, out)
    right = sum(i["true"] == i["pred"] for i in items)
    print(f"{a.split}: {right}/{len(items)} correct ({right / len(items):.1%}); wrote metrics.png"
          + (", segmenter_training.png" if seg_ok else "") + f", {len(sheets)} prediction sheets to {out}")


if __name__ == "__main__":
    main()
