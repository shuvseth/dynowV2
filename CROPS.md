# Sidewalk-crop condition classifier (clear / slush / snow)

Why: the first classifier was trained on dashcam ROAD photos, where dark/brown surface = clear. On sidewalks it called
brown slush "clear 0.98". This set trains on masked SIDEWALK crops labeled by a person.

```bash
python -m dynow_vision.cropset make --images data/raw/seg_images --source "..." --license "..." --group-regex '^(yardseries)_\d+$'
python -m dynow_vision.cropset label --annotator yourname     # opens each crop (original+outline | crop) in Preview
python -m dynow_vision.cropset status
python -m dynow_vision.cropset build                          # c/w/s labels -> data/annotations/crops_manifest.csv (group-aware split)
python -m dynow_vision.train --config configs/crops3.yaml --run-name crops3_v1
python -m dynow_vision.evaluate --config configs/crops3.yaml --checkpoint outputs/runs/crops3_v1/best.pt   # test split, once
python -m dynow_vision.analyze_image --image photo.jpg --checkpoint outputs/runs/crops3_v1/best.pt --seg-model outputs/seg_runs/seg_v1/best
```

More crops: `make --images <folder>` (hand-drawn mask if one exists, otherwise the segmenter's mask, marked "predicted"), or
`make --from-crops outputs/video/<clip>/crops --group <clip>` to reuse crops from a video run (frames of one clip share a group).

Labels: `c` clear (bare surface, dry or wet) | `w` slush (partial cover, patchy snow/ice, dirty melt) | `s` snow (mostly covered) |
`u` unsure | `x` bad crop (mask is mostly road/verge). `u`/`x` are excluded from training. Training refuses without
10/3/3 crops per class in train/val/test. Predicted-mask crops can contain mask errors: mark those `x`.
Keep a held-out set of photos you never add (and report results on it once).
