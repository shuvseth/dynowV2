"""Snow height, two HUMAN-ASSISTED ways. There is no automated snow-depth model in this repo.

1. CATEGORY (relative to a standing person's leg), assigned by a person:
       python -m dynow_vision.snowheight category --annotator alice --images data/raw/some_folder
2. REFERENCE MEASUREMENT: you click a reference object of known size and the snow height in the same photo; the
   height is a pixel ratio, reported with an uncertainty:
       python -m dynow_vision.snowheight measure --annotator alice --image photo.jpg
       python -m dynow_vision.snowheight status

Neither is a validated measurement. The reference method assumes the reference and the snow edge are vertical, at
about the same distance from the camera, with the camera roughly level (see ASSUMPTIONS). Photos without a usable
reference get a category only, never a centimetre value.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .paths import repo_path
from .preprocessing import IMAGE_EXTENSIONS
from .severity import UNKNOWN, consensus, image_id

RUBRIC_VERSION = "0.1-preliminary"
DEFAULT_DIR = "data/annotations/snow_height"
CATEGORIES = {
    1: "None: walking surface visible, no snow cover",
    2: "Below ankle (trace / light cover)",
    3: "Ankle to mid-shin",
    4: "Mid-shin to knee",
    5: "Above knee",
}
SHORT = {1: "none", 2: "below ankle", 3: "ankle-mid shin", 4: "mid shin-knee", 5: "above knee"}
CATEGORY_NOTE = ("Rate the snow ON the walkable path (the deepest typical part). Snowbanks beside the path are not counted. "
                 "u = not assessable. Levels are judged against a standing adult's leg, not measured.")
ASSUMPTIONS = [
    "reference object and snow edge are vertical and about the same distance from the camera",
    "camera held roughly level (no strong tilt up or down)",
    "reference object's VISIBLE height is known and is not itself partly buried in snow",
    "endpoint clicks are accurate to about the stated pixel tolerance",
]
DISCLAIMER = ("Human-assisted and approximate. Not a validated or calibrated snow-depth measurement; "
              "no model estimates snow height.")
MIN_REF_PX = 40            # a reference shorter than this in the image gives meaningless ratios
DEFAULT_TOL_PX = 5.0       # assumed click error per endpoint
CAT_COLUMNS = ["image_id", "filename", "annotator", "height_category", "notes", "timestamp_utc", "rubric_version"]
MEAS_COLUMNS = ["image_id", "filename", "annotator", "reference_name", "reference_cm", "ref_bottom_xy", "ref_top_xy",
                "snow_base_xy", "snow_top_xy", "ref_px", "snow_px", "snow_height_cm", "uncertainty_cm", "tol_px",
                "notes", "timestamp_utc", "method"]
METHOD = "human-clicked reference ratio v1"


# ------------------------------------------------------------------ measurement
def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_height(ref_bottom, ref_top, snow_base, snow_top, ref_cm: float, tol_px: float = DEFAULT_TOL_PX) -> dict:
    """Snow height = (snow pixel length / reference pixel length) * reference real height.
    Uncertainty propagates `tol_px` of click error on every endpoint (independent, 1-sigma-ish)."""
    if ref_cm <= 0:
        raise ValueError("reference height must be > 0 cm")
    ref_px, snow_px = _dist(ref_bottom, ref_top), _dist(snow_base, snow_top)
    if ref_px < MIN_REF_PX:
        raise ValueError(f"reference is only {ref_px:.0f}px tall in the image (< {MIN_REF_PX}px): too small to scale from")
    e = tol_px * math.sqrt(2)                      # error of a distance between two clicked points
    h = snow_px / ref_px * ref_cm
    rel = math.sqrt((e / ref_px) ** 2 + ((e / snow_px) ** 2 if snow_px > 0 else 0.0))
    unc = h * rel if snow_px > 0 else e / ref_px * ref_cm
    return {"snow_height_cm": round(h, 1), "uncertainty_cm": round(unc, 1), "ref_px": round(ref_px, 1),
            "snow_px": round(snow_px, 1), "tol_px": tol_px}


# ------------------------------------------------------------------ storage
def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _check_annotator(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or ""):
        raise ValueError("annotator must be a simple name (letters, digits, _ . -)")
    return name


def category_path(annotator: str, directory=DEFAULT_DIR) -> Path:
    return repo_path(directory) / f"category_{_check_annotator(annotator)}.csv"


def measurements_path(directory=DEFAULT_DIR) -> Path:
    return repo_path(directory) / "measurements.csv"


def read_categories(directory=DEFAULT_DIR) -> list[dict]:
    rows = []
    for f in sorted(repo_path(directory).glob("category_*.csv")):
        for r in _read(f):
            r["height_category"] = UNKNOWN if r["height_category"] == UNKNOWN else int(r["height_category"])
            rows.append(r)
    return rows


def save_category(image: Path, annotator: str, value, notes: str = "", directory=DEFAULT_DIR) -> None:
    if value != UNKNOWN and value not in CATEGORIES:
        raise ValueError(f"category must be 1-5 or {UNKNOWN!r}")
    p = category_path(annotator, directory)
    rows = {r["image_id"]: r for r in _read(p)}
    iid = image_id(image)
    rows[iid] = {"image_id": iid, "filename": image.name, "annotator": annotator, "height_category": value, "notes": notes,
                 "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rubric_version": RUBRIC_VERSION}
    _write(p, CAT_COLUMNS, list(rows.values()))


def save_measurement(image: Path, annotator: str, ref_name: str, ref_cm: float, pts: dict, notes: str = "",
                     tol_px: float = DEFAULT_TOL_PX, directory=DEFAULT_DIR) -> dict:
    res = compute_height(pts["ref_bottom"], pts["ref_top"], pts["snow_base"], pts["snow_top"], ref_cm, tol_px)
    p = measurements_path(directory)
    iid = image_id(image)
    rows = [r for r in _read(p) if not (r["image_id"] == iid and r["annotator"] == annotator)]
    fmt = lambda xy: f"{xy[0]:.0f},{xy[1]:.0f}"
    rows.append({"image_id": iid, "filename": image.name, "annotator": _check_annotator(annotator), "reference_name": ref_name,
                 "reference_cm": ref_cm, "ref_bottom_xy": fmt(pts["ref_bottom"]), "ref_top_xy": fmt(pts["ref_top"]),
                 "snow_base_xy": fmt(pts["snow_base"]), "snow_top_xy": fmt(pts["snow_top"]), **res, "notes": notes,
                 "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "method": METHOD})
    _write(p, MEAS_COLUMNS, rows)
    return res


def display_category(values: list) -> str:
    c = consensus(values)
    if c["status"] in ("single", "agreed"):
        return f"{SHORT[c['value']]} ({c['n_annotators']} annotator{'s' if c['n_annotators'] > 1 else ''})"
    if c["status"] == "close":
        lo, hi = c["range"]
        return f"{SHORT[lo]} to {SHORT[hi]} (annotators differ by 1)"
    return {"unknown": "not assessable", "not_annotated": "not annotated"}.get(c["status"], "DISPUTED between annotators")


def lookup_lines(image_path: str | Path, directory=DEFAULT_DIR) -> list[str]:
    """Human-entered snow height info for an image, as banner lines. Empty if none was recorded."""
    try:
        iid = image_id(image_path)
    except OSError:
        return []
    lines = []
    cats = [r["height_category"] for r in read_categories(directory) if r["image_id"] == iid]
    if cats:
        lines.append(f"snow height (human category): {display_category(cats)}")
    for r in _read(measurements_path(directory)):
        if r["image_id"] == iid:
            lines.append(f"snow height (human-assisted, vs {r['reference_name']}): ~{float(r['snow_height_cm']):.0f} "
                         f"+/- {float(r['uncertainty_cm']):.0f} cm by {r['annotator']}")
    return lines


# ------------------------------------------------------------------ interactive
POINT_PROMPTS = ["1) BOTTOM of the reference object", "2) TOP of the reference object",
                 "3) bare GROUND / path surface level beside the snow edge", "4) TOP of the snow at the same distance"]


def pick_points_gui(path: Path) -> list[tuple[float, float]] | None:
    """Click 4 points in an interactive matplotlib window (right-click or close to cancel)."""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    pts: list[tuple[float, float]] = []
    for i, prompt in enumerate(POINT_PROMPTS):
        ax.set_title(f"{path.name}: click {prompt}   (close window to skip)")
        fig.canvas.draw_idle()
        got = plt.ginput(1, timeout=0)
        if not got:
            plt.close(fig)
            return None
        pts.append(got[0])
        ax.plot(*got[0], "r+", markersize=14)
        if i % 2 == 1:
            ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], "-", color="yellow" if i == 1 else "cyan")
    plt.close(fig)
    return pts


def parse_xy(text: str) -> tuple[float, float]:
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*", text)
    if not m:
        raise ValueError("enter a point as x,y in image pixels")
    return float(m[1]), float(m[2])


def measure_one(image: Path, annotator: str, directory=DEFAULT_DIR, input_fn=input, pick_fn=pick_points_gui, out=print) -> dict | None:
    out(f"\n{image.name}\n  Needs a reference object of KNOWN height in the same photo (hydrant, curb, post, ruler...).")
    out("  Assumptions: " + "; ".join(ASSUMPTIONS))
    name = input_fn("reference object name (Enter = no usable reference, skip) > ").strip()
    if not name:
        return None
    try:
        ref_cm = float(input_fn("reference VISIBLE height in cm > ").strip())
    except ValueError:
        out("not a number; skipped")
        return None
    pts = pick_fn(image)
    if pts is None:
        out("cancelled")
        return None
    if len(pts) != 4:
        raise ValueError("need exactly 4 points")
    d = dict(zip(("ref_bottom", "ref_top", "snow_base", "snow_top"), pts))
    notes = input_fn("notes (optional) > ").strip()
    try:
        res = save_measurement(image, annotator, name, ref_cm, d, notes, directory=directory)
    except ValueError as e:
        out(f"not saved: {e}")
        return None
    out(f"  snow height ~ {res['snow_height_cm']:.0f} +/- {res['uncertainty_cm']:.0f} cm (human-assisted, approximate)")
    return res


def category_session(images: list[Path], annotator: str, directory=DEFAULT_DIR, input_fn=input, show_fn=None, out=print) -> int:
    _check_annotator(annotator)
    done = {r["image_id"] for r in read_categories(directory) if r["annotator"] == annotator}
    queue = [p for p in images if image_id(p) not in done]
    out(f"Rubric {RUBRIC_VERSION}: " + CATEGORY_NOTE)
    for k, v in CATEGORIES.items():
        out(f"  {k}: {v}")
    n = 0
    for i, p in enumerate(queue, 1):
        out(f"\n[{i}/{len(queue)}] {p.name}")
        if show_fn:
            show_fn(p)
        while True:
            ans = input_fn("category [1-5, u=not assessable, Enter=skip, q=quit] > ").strip().lower()
            if ans in ("", "q") or ans in ("u", "unknown") or (ans.isdigit() and int(ans) in CATEGORIES):
                break
            out("enter 1-5, u, Enter or q")
        if ans == "q":
            break
        if ans == "":
            continue
        save_category(p, annotator, UNKNOWN if ans in ("u", "unknown") else int(ans), input_fn("notes (optional) > ").strip(), directory)
        n += 1
    return n


def status(directory=DEFAULT_DIR) -> dict:
    cats = read_categories(directory)
    by_img = defaultdict(list)
    for r in cats:
        by_img[r["image_id"]].append(r["height_category"])
    meas = _read(measurements_path(directory))
    return {"images_with_category": len(by_img), "category_annotators": len({r["annotator"] for r in cats}),
            "images_with_2plus_annotators": sum(len(v) >= 2 for v in by_img.values()),
            "disputed": sum(consensus(v)["status"] == "disputed" for v in by_img.values()),
            "measurements": len(meas), "disclaimer": DISCLAIMER}


def _open_in_preview(path) -> None:
    import subprocess
    subprocess.run(["open", str(path)], check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("category")
    c.add_argument("--annotator", required=True)
    c.add_argument("--images", required=True, help="folder of images")
    c.add_argument("--no-preview", action="store_true")
    m = sub.add_parser("measure")
    m.add_argument("--annotator", required=True)
    m.add_argument("--image", required=True, help="image file or folder")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        import json
        print(json.dumps(status(), indent=2))
        return
    src = Path(a.image if a.cmd == "measure" else a.images)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if src.is_dir() else [src]
    if a.cmd == "category":
        n = category_session(files, a.annotator, show_fn=None if a.no_preview else _open_in_preview)
        print(f"saved {n} categories")
    else:
        for f in files:
            measure_one(f, a.annotator)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
