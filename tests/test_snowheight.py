"""Snow height tests. All values here are synthetic test data, not real measurements."""
import math

import pytest
from PIL import Image

from dynow_vision import snowheight as sh


def test_compute_height_ratio_and_uncertainty():
    r = sh.compute_height((100, 400), (100, 200), (300, 400), (300, 350), ref_cm=60)   # ref 200px = 60cm; snow 50px
    assert r["snow_height_cm"] == 15.0 and r["ref_px"] == 200 and r["snow_px"] == 50
    assert r["uncertainty_cm"] > 0
    far = sh.compute_height((100, 400), (100, 200), (300, 400), (300, 350), 60, tol_px=10)
    assert far["uncertainty_cm"] > r["uncertainty_cm"]                      # sloppier clicks, wider band
    zero = sh.compute_height((100, 400), (100, 200), (300, 400), (300, 400), 60)
    assert zero["snow_height_cm"] == 0 and zero["uncertainty_cm"] > 0       # "no snow" still carries click error


def test_compute_height_refuses_bad_input():
    with pytest.raises(ValueError):
        sh.compute_height((0, 0), (0, 10), (5, 0), (5, 5), 60)              # reference only 10px tall
    with pytest.raises(ValueError):
        sh.compute_height((0, 0), (0, 100), (5, 0), (5, 5), 0)


def _img(tmp_path, name="a.jpg", color=(10, 20, 30)):
    p = tmp_path / name
    Image.new("RGB", (64, 64), color).save(p)
    return p


def test_category_session_consensus_and_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "repo_path", lambda p: tmp_path / p)
    img = _img(tmp_path)
    answers = iter(["3", "", "q"])
    assert sh.category_session([img], "alice", input_fn=lambda _: next(answers), out=lambda *_: None) == 1
    answers = iter(["u", ""])
    sh.category_session([_img(tmp_path, "b.jpg", (1, 2, 3))], "alice", input_fn=lambda _: next(answers), out=lambda *_: None)
    sh.save_category(img, "bob", 4)
    lines = sh.lookup_lines(img)
    assert len(lines) == 1 and "ankle-mid shin to mid shin-knee" in lines[0]    # differ by 1 -> shown as a range
    sh.save_category(img, "carol", 1)
    assert "DISPUTED" in sh.lookup_lines(img)[0]                            # never averaged away
    assert "not assessable" in sh.lookup_lines(tmp_path / "b.jpg")[0]
    with pytest.raises(ValueError):
        sh.save_category(img, "bob", 9)
    with pytest.raises(ValueError):
        sh.category_path("../evil")


def test_measure_flow_saves_and_labels_as_human_assisted(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "repo_path", lambda p: tmp_path / p)
    img = _img(tmp_path)
    ans = iter(["fire hydrant", "70", "check"])
    pts = [(10, 300), (10, 100), (50, 300), (50, 240)]                       # ref 200px = 70cm, snow 60px
    res = sh.measure_one(img, "alice", input_fn=lambda _: next(ans), pick_fn=lambda p: pts, out=lambda *_: None)
    assert res["snow_height_cm"] == 21.0
    line = sh.lookup_lines(img)[0]
    assert "human-assisted" in line and "fire hydrant" in line and "21" in line
    # no reference -> nothing saved, no cm value invented
    assert sh.measure_one(img, "bob", input_fn=lambda _: "", pick_fn=lambda p: pts, out=lambda *_: None) is None
    assert sh.status()["measurements"] == 1
    assert math.isclose(sh.parse_xy("12, 30.5")[1], 30.5)
    with pytest.raises(ValueError):
        sh.parse_xy("abc")
