"""End-to-end mechanics on synthetic images (NOT a statement about real-world accuracy)."""
import json

import pytest

from dynow_vision.dataset import make_splits, read_manifest, write_manifest
from dynow_vision.evaluate import compute_metrics, evaluate
from dynow_vision.predict import predict_image
from dynow_vision.train import load_config, train


@pytest.fixture
def tiny_cfg(workspace):
    write_manifest(make_splits(read_manifest(workspace["manifest"])), workspace["manifest"])
    cfg = load_config("configs/phase1.yaml", {"epochs": 2, "batch_size": 8, "device": "cpu"})
    cfg.update(manifest=str(workspace["manifest"]), raw_dir=str(workspace["raw"]),
               output_root=str(workspace["root"] / "runs"), run_name="t",
               min_per_class={"train": 1, "val": 1, "test": 1})
    return cfg


def test_train_resume_evaluate_predict(tiny_cfg, workspace):
    run = train(tiny_cfg)
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    assert (run / "config.yaml").exists() and (run / "environment.json").exists()
    hist = (run / "history.csv").read_text().splitlines()
    assert len(hist) == 3  # header + 2 epochs

    tiny_cfg["epochs"] = 3                         # resume continues, does not restart
    train(tiny_cfg, str(run / "last.pt"))
    assert len((run / "history.csv").read_text().splitlines()) == 4

    out = workspace["root"] / "eval"
    m = evaluate(run / "best.pt", workspace["manifest"], "test", out, "cpu")
    assert m["preliminary"] is True and m["n_test"] > 0
    assert {"metrics.json", "predictions.csv", "confusion_matrix.png", "examples.png"} <= {p.name for p in out.iterdir()}

    img = workspace["raw"] / read_manifest(workspace["manifest"])[0]["filename"]
    res = predict_image(img, run / "best.pt", "cpu")
    assert res["predicted_class"] in ("clear", "snow") and 0 <= res["model_score"] <= 1
    assert res["inference_time_ms"] > 0 and "not a calibrated" in res["score_note"].lower()
    json.dumps(res)


def test_refuses_tiny_dataset(tiny_cfg):
    tiny_cfg["min_per_class"] = {"train": 10_000, "val": 1, "test": 1}
    with pytest.raises(SystemExit, match="Not enough labeled data"):
        train(tiny_cfg)


def test_predict_rejects_bad_image(tmp_path):
    bad = tmp_path / "x.jpg"; bad.write_bytes(b"junk")
    with pytest.raises(ValueError):
        predict_image(bad, "whatever.pt")


def test_metrics_math():
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["clear", "snow"])
    assert m["accuracy"] == 0.75 and m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["per_class"]["snow"]["precision"] == pytest.approx(2 / 3)
    assert m["preliminary"] is True
