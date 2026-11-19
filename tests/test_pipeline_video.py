"""Video pipeline tests with fake segmenter/classifier on a synthetic video: mechanics only."""
import csv
import json

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from test_segment import ID2LABEL, FakeModel, make_image, seg

from dynow_vision.pipeline import (CLEAR, NO_SIDEWALK, SNOW, UNCERTAIN, UNRELIABLE, Params,
                                   analyze_frame, compare_with_labels, decide_status, make_crop, mask_bbox, run_video)

P = Params()
INFO = {"classifier": {"model_version": "fake"}, "segmenter": {"model": "fake"}}


def fake_clf(label="snow", score=0.9):
    return lambda crop: (label, score, {"clear": 1 - score if label == "snow" else score, "snow": score if label == "snow" else 1 - score})


def make_video(path, n=30, fps=10, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        w.write(np.random.default_rng(i).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    w.release()


def test_decide_status_branches():
    assert decide_status(0.001, 0.9, None, None, P)[0] == NO_SIDEWALK
    assert decide_status(0.2, 0.3, 0.99, "snow", P)[0] == UNRELIABLE
    assert decide_status(0.2, None, None, None, P)[0] == UNRELIABLE
    assert decide_status(0.2, 0.9, 0.55, "snow", P)[0] == UNCERTAIN
    assert decide_status(0.2, 0.9, 0.95, "clear", P)[0] == CLEAR


def test_bbox_and_masked_crop_hides_non_sidewalk():
    img = Image.fromarray(np.full((100, 200, 3), 255, np.uint8))
    mask = np.zeros((100, 200), bool); mask[40:80, 20:100] = True
    assert mask_bbox(np.zeros((5, 5), bool)) is None
    crop, box = make_crop(img, mask, "masked", pad=0.0)
    assert box == [20, 40, 100, 80] and crop.size == (80, 40)
    box_crop, _ = make_crop(img, mask[:, :] | False, "box", pad=0.25)
    masked, _ = make_crop(img, mask, "masked", pad=0.25)
    arr = np.asarray(masked)
    assert (arr[0, 0] != 255).all()          # padding area outside the sidewalk is neutral, not white snow
    assert (np.asarray(box_crop) == 255).all()
    with pytest.raises(ValueError):
        make_crop(img, mask, "weird")


def test_analyze_frame_with_sidewalk_and_without():
    img = make_image(size=(160, 120))
    r = analyze_frame(img, 0, 0.0, seg(), fake_clf("snow", 0.9), Params(min_seg_confidence=0.0))
    assert r.status == SNOW and r.crop is not None and r.bbox_xyxy and r.classification_ms >= 0
    only_road = {0: "road", 1: "sidewalk"}
    class NoSide(torch.nn.Module):
        def forward(self, pixel_values):
            lg = torch.zeros(1, 2, 8, 8); lg[0, 0] = 10
            return type("O", (), {"logits": lg})()
    from dynow_vision.segment import Segmenter
    s = Segmenter(model=NoSide(), id2label=only_road, device="cpu", max_side=128)
    called = []
    r2 = analyze_frame(img, 5, 0.5, s, lambda c: called.append(1) or ("snow", 1.0, {}), P)
    assert r2.status == NO_SIDEWALK and r2.crop is None and not called   # classifier never consulted


def test_road_snow_not_attributed_to_sidewalk():
    """The crop given to the classifier contains only sidewalk pixels (everything else neutral)."""
    img = Image.fromarray(np.full((120, 160, 3), 255, np.uint8))   # all-white 'snow' frame
    seen = []
    analyze_frame(img, 0, 0, seg(), lambda c: seen.append(np.asarray(c)) or ("clear", 0.9, {}), Params(min_seg_confidence=0.0, crop_pad=0.0))
    crop = seen[0].reshape(-1, 3)
    fill = np.array([124, 116, 104])
    is_white = (crop == 255).all(1)
    is_fill = (crop == fill).all(1)
    assert is_white.any() and is_fill.all() | (is_white | is_fill).all()   # only sidewalk pixels or neutral fill
    assert (is_white | is_fill).all()


def test_run_video_end_to_end(tmp_path):
    v = tmp_path / "v.mp4"; make_video(v, n=30, fps=10)
    rep = run_video(v, tmp_path / "out", seg(), fake_clf("clear", 0.95), Params(sample_fps=2, min_seg_confidence=0.0, out_max_side=96), INFO)
    out = tmp_path / "out"
    assert rep["run"]["sampled_frames"] == 6 and rep["run"]["sample_step_frames"] == 5
    assert rep["summary"]["counts"][CLEAR] == 6
    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30 and cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 96
    data = json.loads((out / "predictions.json").read_text())
    f0 = data["frames"][0]
    assert {"frame_index", "timestamp_s", "status", "model_score", "segmentation_ms", "classification_ms"} <= set(f0)
    assert data["models"]["classifier"]["model_version"] == "fake" and data["limitations"]
    assert (out / "masks" / "frame000000_sidewalk.png").exists() and (out / "representative" / "clear.jpg").exists()
    assert (out / "crops" / "frame000000.jpg").exists()


def test_bad_video_path(tmp_path):
    with pytest.raises(ValueError):
        run_video(tmp_path / "nope.mp4", tmp_path / "o", seg(), fake_clf(), P, INFO)


def test_compare_with_labels(tmp_path):
    rep = {"frames": [{"frame_index": 0, "status": SNOW}, {"frame_index": 5, "status": CLEAR},
                      {"frame_index": 10, "status": UNCERTAIN}, {"frame_index": 15, "status": SNOW}]}
    (tmp_path / "p.json").write_text(json.dumps(rep))
    with open(tmp_path / "l.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_index", "label"])
        w.writerows([(0, "snow"), (5, "snow"), (10, "snow"), (15, "uncertain"), (99, "snow")])
    c = compare_with_labels(tmp_path / "p.json", tmp_path / "l.csv")
    assert c["n_labeled_clear_or_snow"] == 3 and c["n_not_in_sampled_frames"] == 1
    assert c["n_answered_by_pipeline"] == 2 and c["accuracy_on_answered"] == 0.5
    assert c["accuracy_counting_abstentions_as_wrong"] == pytest.approx(1 / 3)
