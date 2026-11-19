# Phase 4: preliminary obstruction severity (human-assigned)

**Status: human annotation workflow only. There is no automated severity model**, because no human-labeled data exists yet. Any severity shown in the video/JSON is a human's rating of that frame; frames nobody rated say "not annotated".

> The 1-5 score is a research rubric. It is **not** a validated accessibility assessment or certification, it does not establish that a route is safe or unsafe for any disability group, it is **not** a snow-depth measurement (RGB images cannot give physical depth without scale/depth information), and transparent ice cannot be reliably judged from RGB.

## Rubric v0.1-preliminary (rate the visible pedestrian passage in the whole frame)

| Score | Meaning | Typical visual evidence |
|---|---|---|
| 1 | **Clear** | Walking surface visible and continuous; no meaningful snow cover. |
| 2 | **Minor** | Patches or light snow on part of the surface; a continuous passage is obvious. |
| 3 | **Moderate** | Substantial or uneven/packed snow, but a continuous passage (shoveled strip, trodden path) is visibly present. |
| 4 | **Severe** | Surface mostly covered with no clearly defined cleared passage, OR a snowbank partly blocks the path or a curb ramp. |
| 5 | **Blocked** | No continuous passage visible: e.g. snowbank across the full path width, curb ramp buried. |
| **u** | **Not assessable** | Sidewalk not visible, occluded, too dark/blurry, no sidewalk in view, or you cannot tell. Use it freely; it is not an error. |

Factors to weigh: how much of the walking surface is snow-covered, whether a continuous passage is apparent, snowbanks across the path, visible curb-ramp obstruction, and occluded/uncertain areas (rate `u`, or lower your confidence, rather than guess). Optional flags per image: `snowbank_across_path` (yes/no/not_recorded), `curb_ramp_blocked` (yes/no/not_recorded), free-text notes.

Assumptions and limits: levels are ordinal and the boundaries (e.g. "mostly covered") are subjective; the rubric describes what is *visible in one frame*, not slope, surface hardness, ice, width in cm, or any person's mobility needs. Rubric thresholds are untested until annotators' agreement is measured.

## Workflow

```bash
# 1. export frames at the same indices the video pipeline samples (default 2 fps)
python -m dynow_vision.severity frames --video clip.mp4            # -> data/raw/severity_frames/<clip>/
# 2. each annotator rates the frames (opens each in Preview; saved after every answer)
python -m dynow_vision.severity annotate --annotator alice --images data/raw/severity_frames/clip
#    keys: 1-5, u = not assessable, Enter = skip, b = back, q = quit; then flags and notes
# 3. agreement between annotators who rated the same images
python -m dynow_vision.severity agreement
# 4. counts and whether an automated model would even be justified
python -m dynow_vision.severity status
# 5. show human severity in the annotated video + JSON
python -m dynow_vision.pipeline --video clip.mp4 --checkpoint outputs/runs/udot_proxy_v1/best.pt
```

Annotations are stored **one CSV per annotator** in `data/annotations/severity/<annotator>.csv` (columns: image_id, filename, source_video, frame_index, annotator, severity, snowbank_across_path, curb_ramp_blocked, notes, timestamp_utc). They are separate from all model output (`outputs/`). Each annotator should rate independently, without seeing the others' labels or the model's prediction.

**Several annotators on one frame:** never averaged. Shown as `agreed` (all equal), `close` (differ by 1, shown as a range), `disputed` (differ by 2+, or one says not assessable), or `NOT ASSESSABLE` (all say u).

**Agreement:** `agreement` reports, per annotator pair on shared images: exact agreement, share within one level, number of assessable/not-assessable disagreements, and quadratic-weighted Cohen's kappa (undefined and reported as such when ratings are all identical). Pairs with fewer than 30 shared images carry an explicit "very uncertain" warning. No agreement has been measured yet (no annotations).

## Where severity appears
`annotated.mp4` shows `Severity (human, preliminary): ...` under the model label on every frame (held frames show their sample frame's value) plus a footer saying it is not a certification. `predictions.json` has per-frame `severity` {status, value, range, n_annotators, display} and a top-level `severity` block (rubric version, source, disclaimer, annotated/not-annotated counts). Only the exact sampled frames that were annotated carry a rating.

## Automated severity model: not built
Training one needs real data. `severity status` gates this with judgment-call thresholds (>=150 images with an agreed level, >=15 per level, annotator kappa >= 0.6). Current state: 0 annotated images, so **no model was trained and none is claimed**. If those thresholds are ever met, a small head on frozen DINOv2 features (as in Phase 1) with group-aware held-out evaluation would be the first thing to try. Until then severity stays a human-assigned assessment.

## Verification done
- 44 tests pass (rubric parsing, per-annotator storage, resume/skip, invalid input, consensus rules, agreement math incl. undefined kappa, frame export aligned with pipeline sampling, severity in JSON and video).
- Rendering was checked on a real clip (53 s plaza video, 300 frames): "not annotated", agreed, close and disputed states all display correctly. The non-empty states in that check used **invented placeholder values kept outside the repo** purely to test display; they are not assessments and were not saved.
