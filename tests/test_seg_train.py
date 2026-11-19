import json

import numpy as np
import pytest
import torch
from PIL import Image

from dynow_vision import seg_data, seg_train
from dynow_vision.seg_data import shapes_to_mask, split_groups


def test_shapes_to_mask_polygon_rect_and_unlabelled():
    m = shapes_to_mask([{"label": "sidewalk", "points": [[2, 2], [8, 2], [8, 8], [2, 8]], "shape_type": "polygon"}], 10, 10)
    assert m.sum() > 20 and m[5, 5] == 1 and m[0, 0] == 0
    r = shapes_to_mask([{"label": "Sidewalk", "points": [[0, 0], [5, 5]], "shape_type": "rectangle"}], 10, 10)
    assert r[2, 2] == 1 and r[9, 9] == 0
    assert shapes_to_mask([], 10, 10) is None                       # unlabelled is NOT "no sidewalk"
    assert shapes_to_mask([{"label": "no_sidewalk", "points": [[0, 0]], "shape_type": "point"}], 10, 10).sum() == 0
    assert shapes_to_mask([{"label": "road", "points": [[0, 0], [5, 0], [5, 5]]}], 10, 10) is None


def test_split_groups_keeps_groups_together():
    stems = [f"a_{i}" for i in range(6)] + [f"b_{i}" for i in range(6)] + [f"c{i}" for i in range(8)]
    groups = {s: s.split("_")[0] if "_" in s else s for s in stems}
    sp = split_groups(stems, groups, seed=1)
    for g in ("a", "b"):
        assert len({sp[s] for s in stems if groups[s] == g}) == 1
    assert set(sp.values()) <= {"train", "val", "test"}
    assert sp == split_groups(stems, groups, seed=1)


def _make_project(tmp_path, monkeypatch, n=8):
    monkeypatch.setattr(seg_data, "repo_path", lambda p: tmp_path / p if not str(p).startswith("/") else __import__("pathlib").Path(p))
    monkeypatch.setattr(seg_train, "repo_path", seg_data.repo_path)
    imgs = tmp_path / "data/raw/seg_images"
    imgs.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(imgs / f"p{i}.jpg")
        (imgs / f"p{i}.json").write_text(json.dumps({"imageWidth": 128, "imageHeight": 96, "shapes": [
            {"label": "sidewalk", "points": [[30, 40], [100, 40], [100, 95], [30, 95]], "shape_type": "polygon"}]}))


def test_prepare_and_tiny_fine_tune(tmp_path, monkeypatch):
    _make_project(tmp_path, monkeypatch)
    r = seg_data.prepare("unit test", "own work")
    assert r["n_labelled"] == 8 and sum(r["splits"].values()) == 8
    rows = seg_data.read_manifest()
    assert all(x["license"] == "own work" for x in rows)
    with pytest.raises(ValueError):
        seg_data.prepare("", "x")
    with pytest.raises(SystemExit):                                  # too little data refuses to train
        seg_train.check_enough(rows, allow_tiny=False)
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    cfg = SegformerConfig(num_labels=2, id2label=seg_train.ID2LABEL, hidden_sizes=[8, 16, 32, 32], depths=[1, 1, 1, 1],
                          num_attention_heads=[1, 1, 2, 2], decoder_hidden_size=16)
    torch.manual_seed(0)
    model = SegformerForSemanticSegmentation(cfg)
    out = tmp_path / "run"
    res = seg_train.fine_tune(rows, out, epochs=2, batch_size=2, device="cpu", crop=64, long_side=128, model=model, baseline=False)
    assert len(res["history"]) == 2 and (out / "best" / "config.json").exists() and (out / "metrics.json").exists()
    assert res["best_val"]["iou_pooled"] is not None
