"""Crop-set tests. Images and labels here are synthetic."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dynow_vision import cropset
from dynow_vision.dataset import read_manifest, validate_rows


@pytest.fixture()
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr(cropset, "repo_path", lambda p: tmp_path / p)
    monkeypatch.setattr("dynow_vision.dataset.repo_path", lambda p: tmp_path / p if not str(p).startswith("/") else __import__("pathlib").Path(p))
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rng = np.random.default_rng(1)
    for i in range(12):
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(imgs / f"p{i:02d}.jpg")
    return tmp_path, sorted(imgs.glob("*.jpg"))


def stub_seg():
    def segment(img):
        m = np.zeros((img.height, img.width), bool)
        m[40:110, 30:130] = True
        return SimpleNamespace(sidewalk_mask=m)
    return SimpleNamespace(segment=segment, model_name="stub-seg")


def test_make_label_build_and_train_ready(proj, monkeypatch):
    tmp, files = proj
    rows = cropset.make_crops(files, "unit test", "own work", seg=stub_seg(), masks_dir="nomasks")
    assert len(rows) == 12 and all(r["mask_source"] == "predicted:stub-seg" for r in rows)
    assert (tmp / cropset.CROPS_DIR / "p00.png").exists() and (tmp / cropset.CROPS_DIR / "preview" / "p00.jpg").exists()
    with pytest.raises(ValueError):
        cropset.make_crops(files, "", "x", seg=stub_seg())

    answers = iter(["c", "w", "s", "u", "x", "zz", "s", "c", "w", "s", "c", "w", "", "q"])
    n = cropset.label_session("tester", input_fn=lambda _: next(answers), out=lambda *_: None)
    labels = cropset.read_labels()
    assert n == len(labels) == 11 and labels["p00.png"]["label"] == "clear" and labels["p03.png"]["label"] == "uncertain"
    with pytest.raises(ValueError):
        cropset.label_session("../evil", input_fn=lambda _: "q")

    res = cropset.build_manifest(seed=1)
    assert res["n"] == 9                                           # u and x are excluded from training
    mrows = read_manifest(cropset.MANIFEST_CSV)
    assert {r["label"] for r in mrows} <= set(cropset.CLASSES)
    assert all(r["filename"].startswith("sidewalk_crops/") and r["license"] == "own work" for r in mrows)
    assert validate_rows(mrows, raw_dir="data/raw") == []          # a third class passes manifest validation
    assert cropset.status()["labelled"] == 11 and cropset.status()["predicted_masks"] == 12


def test_human_mask_is_preferred(proj):
    tmp, files = proj
    mdir = tmp / "masks"
    mdir.mkdir()
    m = np.zeros((120, 160), np.uint8)
    m[10:60, 10:60] = 1
    Image.fromarray(m).save(mdir / "p00.png")
    rows = cropset.make_crops(files[:2], "t", "own work", seg=stub_seg(), masks_dir="masks")
    assert {r["crop"]: r["mask_source"] for r in rows} == {"p00.png": "human", "p01.png": "predicted:stub-seg"}
    assert Image.open(tmp / cropset.CROPS_DIR / "p00.png").size[0] < 70   # cropped to the human mask box (+pad)
