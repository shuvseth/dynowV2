"""Severity tests. All annotations here are synthetic test data, not real human assessments."""
import csv
import json

import cv2
import numpy as np
import pytest
from test_pipeline_video import INFO, fake_clf, make_video
from test_segment import make_image, seg

from dynow_vision.pipeline import Params, run_video
from dynow_vision.severity import (COLUMNS, LEVELS, UNKNOWN, agreement, annotate_session, consensus,
                                   consensus_by_frame, export_frames, image_id, parse_frame_name,
                                   parse_severity, read_annotations, status, annotator_file)


def imgs(tmp_path, n=3, prefix="clip"):
    d = tmp_path / "imgs"; d.mkdir(exist_ok=True)
    ps = []
    for i in range(n):
        p = d / f"{prefix}_f{i * 15:06d}.png"; make_image(p, size=(96 + i, 80)); ps.append(p)
    return ps


def script(answers):
    it = iter(answers)
    return lambda _="": next(it)


def test_rubric_levels_complete():
    assert list(LEVELS) == [1, 2, 3, 4, 5]
    assert parse_severity("u") == UNKNOWN and parse_severity("3") == 3
    for bad in ("0", "6", "x", ""):
        with pytest.raises(ValueError):
            parse_severity(bad)


def test_annotation_session_saves_separately_per_annotator(tmp_path):
    ps = imgs(tmp_path)
    d = tmp_path / "ann"
    # img1: severity 3 with flags + note; img2: not assessable; img3: skipped; then quit unused
    n = annotate_session(ps, "alice", d, input_fn=script(["3", "s", "icy edge", "u", "", ""]), out=lambda *_: None)
    assert n == 2
    rows = read_annotations(d)
    assert [r["severity"] for r in rows] == [3, UNKNOWN]
    assert rows[0]["snowbank_across_path"] == "yes" and rows[0]["notes"] == "icy edge"
    assert rows[1]["snowbank_across_path"] == "not_recorded"
    assert rows[0]["source_video"] == "clip" and rows[0]["frame_index"] == "0"
    assert annotator_file("alice", d).exists() and not annotator_file("bob", d).exists()
    with open(annotator_file("alice", d)) as f:
        assert next(csv.reader(f)) == COLUMNS


def test_annotation_resume_skips_done_and_validates_input(tmp_path):
    ps = imgs(tmp_path, 2); d = tmp_path / "ann"
    annotate_session(ps, "alice", d, input_fn=script(["2", "n", "", "q"]), out=lambda *_: None)
    msgs = []
    n = annotate_session(ps, "alice", d, input_fn=script(["9", "4", "", "", ]), out=msgs.append)
    assert n == 1 and any("severity must be" in m for m in msgs)
    assert [r["severity"] for r in read_annotations(d)] == [2, 4]


def test_bad_annotator_name_rejected(tmp_path):
    with pytest.raises(ValueError):
        annotator_file("../evil", tmp_path)


def test_consensus_never_hides_disagreement():
    assert consensus([])["status"] == "not_annotated"
    assert consensus([3])["status"] == "single"
    assert consensus([3, 3])["status"] == "agreed"
    assert consensus([3, 4])["status"] == "close" and consensus([3, 4])["value"] is None
    assert consensus([1, 4])["status"] == "disputed"
    assert consensus([UNKNOWN, UNKNOWN])["status"] == "unknown"
    assert consensus([2, UNKNOWN])["status"] == "disputed"


def row(ann, img, sev, video="clip", idx="0"):
    return {"annotator": ann, "image_id": img, "severity": sev, "source_video": video, "frame_index": idx}


def test_agreement_metrics():
    a = [row("a", str(i), s) for i, s in enumerate([1, 2, 3, 4, 5, 3])]
    b = [row("b", str(i), s) for i, s in enumerate([1, 2, 4, 4, 5, UNKNOWN])]
    r = agreement(a + b)
    p = r["pairs"][0]
    assert p["n_shared"] == 6 and p["assessable_disagreement"] == 1 and p["n_both_assessable"] == 5
    assert p["exact_agreement_all"] == pytest.approx(4 / 6) and p["within_one_level"] == 1.0
    assert 0.8 < p["quadratic_weighted_kappa"] <= 1.0 and "warning" in p       # tiny n flagged
    assert "Need at least 2" in agreement(a)["notes"][0]
    same = agreement([row("a", "1", 3), row("b", "1", 3)])["pairs"][0]
    assert same["quadratic_weighted_kappa"] is None                             # undefined, not faked


def test_status_blocks_model_training_when_data_is_small():
    s = status([row("a", str(i), 3) for i in range(10)])
    assert s["model_training_eligible"] is False and s["blocking_reasons"]
    assert "No automated severity model" in s["note"]


def test_frame_export_matches_pipeline_sampling(tmp_path):
    v = tmp_path / "vid.mp4"; make_video(v, n=30, fps=10)
    saved = export_frames(v, tmp_path / "fr", sample_fps=2)
    assert [parse_frame_name(p.name) for p in saved] == [("vid", str(i)) for i in range(0, 30, 5)]
    assert len({image_id(p) for p in saved}) == len(saved)


def test_pipeline_shows_human_severity_and_not_annotated(tmp_path):
    v = tmp_path / "vid.mp4"; make_video(v, n=30, fps=10)
    ann = [row("a", "x1", 4, "vid", "0"), row("b", "x1", 4, "vid", "0"), row("a", "x2", UNKNOWN, "vid", "10"),
           row("a", "x3", 2, "other_video", "5")]
    sev = consensus_by_frame(ann, "vid")
    rep = run_video(v, tmp_path / "o", seg(), fake_clf("clear", 0.9),
                    Params(sample_fps=2, min_seg_confidence=0.0, out_max_side=96), INFO, severity_by_frame=sev)
    by = {f["frame_index"]: f["severity"] for f in rep["frames"]}
    assert by[0]["status"] == "agreed" and by[0]["value"] == 4 and by[0]["n_annotators"] == 2
    assert by[10]["status"] == "unknown"
    assert by[5]["status"] == "not_annotated" and by[5]["value"] is None   # no invented severity
    assert rep["severity"]["frames_annotated"] == 2 and rep["severity"]["frames_not_annotated"] == 4
    assert "no automated severity model" in rep["severity"]["source"]
    assert any("HUMAN-assigned" in l for l in rep["limitations"])
    assert json.loads((tmp_path / "o" / "predictions.json").read_text())["severity"]["rubric_version"]
    assert cv2.VideoCapture(str(tmp_path / "o" / "annotated.mp4")).isOpened()
