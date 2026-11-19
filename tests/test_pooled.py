"""Mask-pooled classifier tests with a fake backbone (no weights needed). Data is synthetic."""
import numpy as np
import pytest
import torch
from PIL import Image
from test_pipeline_video import make_image, seg  # noqa: F401  (fixtures/helpers)

from dynow_vision import pooled
from dynow_vision.pipeline import Params, analyze_frame

D = 4


class FakeBackbone:
    """Patch (row, col) -> token [col/G, row/G, 1, 0]: left/right and top/bottom patches differ."""
    def forward_features(self, x):
        g = x.shape[-1] // 14
        rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        tok = np.stack([cc / g, rr / g, np.ones_like(rr), np.zeros_like(rr)], -1).reshape(1, g * g, D).astype(np.float32)
        return {"x_norm_patchtokens": torch.from_numpy(tok)}


def payload(classes=("clear", "slush", "snow")):
    coef = np.zeros((3, D), np.float32)
    coef[0, 0], coef[2, 0] = -8, 8                      # token x-position: left -> clear, right -> snow
    return {"type": pooled.CHECKPOINT_TYPE, "classes": list(classes), "backbone": "fake", "input_size": 448, "mean": np.zeros(D, np.float32),
            "std": np.ones(D, np.float32), "coef": coef, "intercept": np.array([4, 0, -4], np.float32), "run_name": "t", "trained_by": "test"}


def test_features_come_from_the_masked_region_only():
    img = Image.new("RGB", (200, 100), (120, 120, 120))
    left = np.zeros((100, 200), bool); left[:, :50] = True
    right = np.zeros((100, 200), bool); right[:, 150:] = True
    fl, nl = pooled.extract_features(FakeBackbone(), img, left, "cpu")
    fr, nr = pooled.extract_features(FakeBackbone(), img, right, "cpu")
    assert fl[0] < 0.3 < 0.7 < fr[0] and nl > 3 and nr > 3
    tiny = np.zeros((100, 200), bool); tiny[50, 100] = True
    assert pooled.extract_features(FakeBackbone(), img, tiny, "cpu")[1] >= pooled.MIN_PATCHES    # tiny mask still pools >= 3 patches


def test_classifier_scores_are_probabilities_and_follow_the_mask():
    clf = pooled.PooledClassifier(FakeBackbone(), payload(), "cpu")
    img = Image.new("RGB", (200, 100))
    left = np.zeros((100, 200), bool); left[:, :50] = True
    right = np.zeros((100, 200), bool); right[:, 150:] = True
    c1, s1, d1 = clf(None, image=img, mask=left)
    c2, s2, d2 = clf(None, image=img, mask=right)
    assert (c1, c2) == ("clear", "snow") and abs(sum(d1.values()) - 1) < 1e-6 and 0 < s1 <= 1
    with pytest.raises(ValueError):
        clf(None)                                       # refuses to guess without image + mask


def test_checkpoint_roundtrip_and_pipeline_integration(tmp_path):
    path = tmp_path / "best.pt"
    torch.save(payload(), path)
    assert pooled.is_pooled_checkpoint(path) and not pooled.is_pooled_checkpoint(tmp_path / "missing.pt")
    clf, pl = pooled.load_pooled(path, "cpu", backbone=FakeBackbone())
    assert pl["checkpoint_sha256"] and clf.classes == ["clear", "slush", "snow"]
    r = analyze_frame(make_image(size=(160, 120)), 0, 0.0, seg(), clf, Params(min_seg_confidence=0.0))
    assert r.status in ("clear", "slush", "snow", "uncertain", "no_sidewalk", "unreliable_sidewalk")
    if r.predicted_class:
        assert r.class_scores and abs(sum(r.class_scores.values()) - 1) < 1e-6
