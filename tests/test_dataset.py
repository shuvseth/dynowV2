import pytest
from PIL import Image

from dynow_vision.dataset import (ManifestError, WinterDataset, add_images, check_split_integrity,
                                  class_summary, make_splits, read_manifest, validate_rows, write_manifest)
from dynow_vision.label import apply_label, label_session
from dynow_vision.preprocessing import eval_transform, train_transform, validate_image


def test_add_images_registers_group_and_skips_duplicates(workspace):
    rows = read_manifest(workspace["manifest"])
    assert len(rows) == 48
    assert {r["group"] for r in rows if r["filename"].startswith("snow0")} == {"snow0"}
    again = add_images([workspace["src"]], "x", "y", group_regex=r"^(\w+?)_f\d+$",
                       manifest=workspace["manifest"], raw_dir=workspace["raw"])
    assert again["added"] == [] and len(again["skipped_existing"]) == 48


def test_invalid_images_rejected(tmp_path):
    bad = tmp_path / "broken.jpg"; bad.write_bytes(b"not an image")
    tiny = tmp_path / "tiny.png"; Image.new("RGB", (10, 10)).save(tiny)
    txt = tmp_path / "a.txt"; txt.write_text("hi")
    for p in (bad, tiny, txt, tmp_path / "missing.png"):
        assert validate_image(p)[0] is False
    res = add_images([bad, tiny], "s", "l", manifest=tmp_path / "m.csv", raw_dir=tmp_path / "raw")
    assert res["added"] == [] and len(res["rejected"]) == 2


def test_add_requires_source_and_license(tmp_path):
    with pytest.raises(ManifestError):
        add_images([tmp_path], "", "", manifest=tmp_path / "m.csv", raw_dir=tmp_path / "raw")


def test_label_validation(workspace):
    rows = read_manifest(workspace["manifest"])
    assert validate_rows(rows, workspace["raw"]) == []
    rows[0]["label"] = "icy"
    rows[1]["source"] = ""
    msgs = "\n".join(validate_rows(rows, workspace["raw"]))
    assert "invalid label 'icy'" in msgs and "missing 'source'" in msgs
    with pytest.raises(ValueError):
        apply_label(rows[0], "banana")


def test_missing_file_reported(workspace):
    (workspace["raw"] / read_manifest(workspace["manifest"])[0]["filename"]).unlink()
    assert any("does not exist" in p for p in validate_rows(read_manifest(workspace["manifest"]), workspace["raw"]))


def test_split_integrity_and_determinism(workspace):
    rows = make_splits(read_manifest(workspace["manifest"]), seed=7)
    check_split_integrity(rows)
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())          # no frame leakage
    summary = class_summary(rows)
    for sp in ("train", "val", "test"):                          # every split has both classes
        assert summary[sp].get("clear", 0) > 0 and summary[sp].get("snow", 0) > 0
    again = make_splits(read_manifest(workspace["manifest"]), seed=7)
    assert [r["split"] for r in rows] == [r["split"] for r in again]


def test_leakage_detected():
    rows = [{"filename": "a", "source": "s", "license": "l", "group": "g", "label": "snow", "split": "train"},
            {"filename": "b", "source": "s", "license": "l", "group": "g", "label": "snow", "split": "test"}]
    with pytest.raises(ManifestError):
        check_split_integrity(rows)


def test_uncertain_excluded_from_splits(workspace):
    rows = read_manifest(workspace["manifest"])
    rows[0]["label"] = "uncertain"
    rows = make_splits(rows)
    assert rows[0]["split"] == ""


def test_dataset_loading(workspace):
    rows = make_splits(read_manifest(workspace["manifest"]))
    ds = WinterDataset(rows, "train", train_transform(224), workspace["raw"])
    x, y, fn = ds[0]
    assert x.shape == (3, 224, 224) and y in (0, 1) and fn.endswith(".png")
    assert ds.items and all(r["split"] == "train" for r in ds.items)
    assert eval_transform(224)(Image.new("RGB", (300, 200))).shape == (3, 224, 224)
    with pytest.raises(ValueError):
        eval_transform(225)


def test_label_session_scripted(workspace):
    rows = read_manifest(workspace["manifest"])
    for r in rows:
        r["label"] = ""
    answers = iter(["c", "s", "u", "", "q"])
    n = label_session(rows, workspace["manifest"], workspace["raw"], input_fn=lambda _: next(answers), out=lambda *_: None)
    saved = read_manifest(workspace["manifest"])
    assert n == 3 and [r["label"] for r in saved[:4]] == ["clear", "snow", "uncertain", ""]


def test_label_map_from_folders(tmp_path):
    from conftest import make_img
    for cls in ("Snow", "Clear", "Wet"):
        d = tmp_path / "in" / "train" / cls; d.mkdir(parents=True)
        make_img(d / f"{cls}1.png", "snow", 1)
    res = add_images([tmp_path / "in"], "s", "l", manifest=tmp_path / "m.csv", raw_dir=tmp_path / "raw",
                     label_map={"Snow": "snow", "Clear": "clear"})
    rows = read_manifest(tmp_path / "m.csv")
    assert {r["label"] for r in rows} == {"snow", "clear"} and len(res["skipped_unmapped"]) == 1
    with pytest.raises(ManifestError):
        add_images([tmp_path / "in"], "s", "l", manifest=tmp_path / "m2.csv", raw_dir=tmp_path / "raw", label_map={"a": "icy"})
