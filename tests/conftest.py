"""Fixtures. Images here are synthetic and exist ONLY to test pipeline mechanics;
they are never used to train or report anything about real sidewalks."""
import numpy as np
import pytest
from PIL import Image

from dynow_vision.dataset import add_images, read_manifest, write_manifest


def make_img(path, kind, seed):
    rng = np.random.default_rng(seed)
    base = 235 if kind == "snow" else 90
    arr = np.clip(rng.normal(base, 12, (96, 128, 3)), 0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


@pytest.fixture
def workspace(tmp_path):
    """Tmp raw dir + manifest with 6 groups x 4 frames per class, all labeled."""
    src = tmp_path / "incoming"; src.mkdir()
    for kind in ("clear", "snow"):
        for g in range(6):
            for f in range(4):
                make_img(src / f"{kind}{g}_f{f}.png", kind, hash((kind, g, f)) % 1000)
    manifest, raw = tmp_path / "manifest.csv", tmp_path / "raw"
    add_images([src], source="synthetic-test", license="n/a", group_regex=r"^(\w+?)_f\d+$",
               manifest=manifest, raw_dir=raw)
    rows = read_manifest(manifest)
    for r in rows:
        r["label"] = "snow" if r["filename"].startswith("snow") else "clear"
    write_manifest(rows, manifest)
    return {"manifest": manifest, "raw": raw, "src": src, "root": tmp_path}
