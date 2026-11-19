"""Segmentation tests. Synthetic images/logits only exercise mechanics; they say nothing about
real-world segmentation quality (see README for the inspected real-image results)."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from dynow_vision.segment import (Segmenter, class_ids, collect_images, colorize, logits_to_result,
                                  make_overlay, process, save_outputs, sidewalk_focus, summarize)

ID2LABEL = {0: "road", 1: "sidewalk", 2: "building", 3: "sky"}
SHUFFLED = {0: "sky", 1: "building", 2: "sidewalk", 3: "road"}  # same names, different ids


class FakeModel(torch.nn.Module):
    """Predicts 'road' in the bottom half, 'sidewalk' in a lower-left block, 'sky' on top."""
    def __init__(self, id2label=ID2LABEL):
        super().__init__()
        self.id2label = id2label
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values):
        _, _, H, W = pixel_values.shape
        h, w = H // 4, W // 4
        names = {v: k for k, v in self.id2label.items()}
        logits = torch.full((1, len(self.id2label), h, w), -10.0)
        logits[0, names["sky"], : h // 2] = 10.0
        logits[0, names["road"], h // 2:] = 10.0
        logits[0, names["sidewalk"], h // 2:, : w // 3] = 12.0
        return type("Out", (), {"logits": logits})()


def seg(id2label=ID2LABEL):
    return Segmenter(model=FakeModel(id2label), id2label=id2label, device="cpu", max_side=256)


def make_image(path=None, size=(200, 150)):
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    if path:
        img.save(path)
    return img


def test_class_mapping_by_name_not_index():
    assert class_ids(ID2LABEL, "road") == [0] and class_ids(SHUFFLED, "road") == [3]
    assert class_ids(SHUFFLED, "sidewalk") == [2]
    with pytest.raises(KeyError):
        class_ids({0: "sky"}, "road")


@pytest.mark.parametrize("labels", [ID2LABEL, SHUFFLED])
def test_masks_match_image_size_and_are_disjoint(labels):
    img = make_image(size=(200, 150))
    res = seg(labels).segment(img)
    assert res.road_mask.shape == res.sidewalk_mask.shape == (150, 200)  # (H, W)
    assert res.road_mask.dtype == bool and res.confidence.shape == (150, 200)
    assert res.road_mask.any() and res.sidewalk_mask.any()
    assert not (res.road_mask & res.sidewalk_mask).any()
    assert res.sidewalk_mask[-1, 0] and res.road_mask[-1, -1] and not res.road_mask[0, 0]


def test_logits_resolution_independent():
    img = make_image(size=(64, 48))
    res = logits_to_result(torch.randn(1, 4, 7, 9), img, ID2LABEL)
    assert res.label_map.shape == (48, 64) and 0 < res.confidence.min() <= res.confidence.max() <= 1


def test_colorize_overlay_and_focus():
    img = make_image()
    res = seg().segment(img)
    color = colorize(res.label_map, ID2LABEL)
    assert color.shape == (150, 200, 3) and tuple(color[-1, -1]) == (128, 64, 128)  # road colour
    assert make_overlay(img, res.label_map, ID2LABEL).size == img.size
    focus = np.asarray(sidewalk_focus(img, res.sidewalk_mask))
    assert (focus[~res.sidewalk_mask] == 0).all()
    assert (focus[res.sidewalk_mask] == np.asarray(img)[res.sidewalk_mask]).all()


def test_save_outputs(tmp_path):
    res = seg().segment(make_image())
    summary = save_outputs(res, tmp_path / "out", {"model": "fake"})
    names = {p.name for p in (tmp_path / "out").iterdir()}
    assert names == {"original.png", "overlay.png", "road_mask.png", "sidewalk_mask.png",
                     "sidewalk_only.png", "confidence.png", "summary.json"}
    mask = np.asarray(Image.open(tmp_path / "out" / "sidewalk_mask.png"))
    assert set(np.unique(mask)) <= {0, 255} and mask.shape == (150, 200)
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["model"] == "fake"
    assert "Not trained by the Dynow team" in summary["model_note"]


def test_no_sidewalk_is_reported_not_invented():
    only_road = {0: "road", 1: "sidewalk"}
    logits = torch.zeros(1, 2, 8, 8); logits[0, 0] = 10
    res = logits_to_result(logits, make_image(), only_road)
    s = summarize(res)
    assert s["sidewalk"]["fraction_of_image"] == 0 and s["sidewalk"]["mean_confidence"] is None
    assert any("No (or almost no) sidewalk" in w for w in s["warnings"])


def test_low_confidence_warning():
    logits = torch.zeros(1, 2, 8, 8); logits[0, 1] = 0.05   # sidewalk wins by a hair -> conf ~0.51
    logits[0, 0, :4] = 5.0                                  # road elsewhere
    res = logits_to_result(logits, make_image(), {0: "road", 1: "sidewalk"})
    assert any("Low model confidence" in w for w in summarize(res)["warnings"])


def test_cli_process_directory_skips_invalid(tmp_path):
    d = tmp_path / "in"; d.mkdir()
    make_image(d / "a.png"); make_image(d / "b.jpg")
    (d / "bad.jpg").write_bytes(b"junk")
    imgs = collect_images(d)
    assert len(imgs) == 3
    results = process(imgs, tmp_path / "out", seg())
    assert sum("error" in r for r in results) == 1
    assert (tmp_path / "out" / "a" / "sidewalk_mask.png").exists() and (tmp_path / "out" / "b" / "road_mask.png").exists()
    assert not (tmp_path / "out" / "bad").exists()


def test_real_pretrained_model_runs_and_maps_classes():
    """Loads the real Cityscapes SegFormer (cached after first download). Synthetic input: shapes/mapping only."""
    transformers = pytest.importorskip("transformers")
    try:
        s = Segmenter(device="cpu", max_side=256)
    except OSError:
        pytest.skip("pretrained segmentation weights not available offline")
    assert s.id2label[0] == "road" and s.id2label[1] == "sidewalk"
    res = s.segment(make_image(size=(320, 240)))
    assert res.label_map.shape == (240, 320) and res.label_map.max() < 19
