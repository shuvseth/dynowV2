"""Local CLI labeling tool.

Shows each unlabeled image in the macOS Preview app (via `open`) and asks for a label.
Labels are written to the manifest CSV after every answer, so you can quit any time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .dataset import DEFAULT_MANIFEST, DEFAULT_RAW, VALID_LABELS, read_manifest, validate_rows, write_manifest
from .paths import repo_path

KEYS = {"c": "clear", "s": "snow", "u": "uncertain"}
HELP = "[c]lear  [s]now  [u]ncertain  [Enter]=skip  [b]ack  [q]uit"


def apply_label(row: dict, label: str) -> None:
    if label not in VALID_LABELS:
        raise ValueError(f"invalid label {label!r}; allowed {VALID_LABELS}")
    row["label"] = label
    row["split"] = ""  # labels changed -> splits must be regenerated


def label_session(rows: list[dict], manifest, raw_dir=DEFAULT_RAW, relabel=False,
                  input_fn=input, show_fn=None, out=print) -> int:
    """Interactive loop. Returns number of labels set/changed."""
    raw = repo_path(raw_dir)
    queue = [i for i, r in enumerate(rows) if relabel or not r["label"]]
    changed, pos = 0, 0
    while pos < len(queue):
        row = rows[queue[pos]]
        out(f"\n[{pos + 1}/{len(queue)}] {row['filename']}   (group={row['group']}, current={row['label'] or '-'})")
        if show_fn:
            show_fn(raw / row["filename"])
        ans = input_fn(f"label {HELP} > ").strip().lower()
        if ans == "q":
            break
        if ans == "b":
            pos = max(0, pos - 1)
            continue
        if ans in KEYS:
            apply_label(row, KEYS[ans])
            write_manifest(rows, manifest)
            changed += 1
        elif ans != "":
            out(f"unrecognised input {ans!r}")
            continue
        pos += 1
    return changed


def _open_in_preview(path) -> None:
    subprocess.run(["open", str(path)], check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dynow_vision.label", description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    ap.add_argument("--relabel", action="store_true", help="also revisit already-labeled images")
    ap.add_argument("--no-open", action="store_true", help="do not open images in Preview")
    args = ap.parse_args(argv)
    rows = read_manifest(args.manifest)
    if not rows:
        print("Manifest is empty. Add images first:  python -m dynow_vision.dataset add <dir> --source ... --license ...")
        return 1
    bad = [p for p in validate_rows(rows, args.raw_dir) if "invalid label" in p or "duplicate" in p]
    if bad:
        print("\n".join(bad)); return 1
    n = label_session(rows, args.manifest, args.raw_dir, args.relabel,
                      show_fn=None if args.no_open else _open_in_preview)
    print(f"\n{n} label(s) saved to {repo_path(args.manifest)}")
    print("Next: python -m dynow_vision.dataset split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
