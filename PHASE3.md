# Phase 3: video pipeline results (2026-11-19)

**Models:** segmentation `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` (pretrained, unmodified); classifier `dinov2_vits14` + head from run `udot_proxy_v1` (frozen DINOv2 pretrained by Meta; head trained by us on UDOT **road** images). Device: Apple MPS. Sampling: 2 fps. Thresholds: sidewalk >= 3% of frame, mean seg confidence >= 0.6, classifier score >= 0.7 (all untuned defaults).

**Footage:** three stock winter videos (1080x1920, ~30 fps) from the local Downloads folder. No project-labeled video exists yet.

## Commands used
```bash
python -m dynow_vision.pipeline --video ~/Downloads/19770687-hd_1080_1920_30fps.mp4 --checkpoint outputs/runs/udot_proxy_v1/best.pt
python -m dynow_vision.pipeline --video ~/Downloads/20794791-hd_1080_1920_30fps.mp4 --checkpoint outputs/runs/udot_proxy_v1/best.pt
python -m dynow_vision.pipeline --video ~/Downloads/6511420-hd_1080_1920_30fps.mp4  --checkpoint outputs/runs/udot_proxy_v1/best.pt
```
Outputs: `outputs/video/<name>/annotated.mp4` etc. (git-ignored).

## Measured results (counts of sampled frames; no ground truth)

| Video | Content | Sampled | snow | clear | uncertain | unreliable | no sidewalk | Sidewalk % of frame (median) |
|---|---|---|---|---|---|---|---|---|
| 19770687 (5.5 s) | icy footpath beside road | 11 | 0 | 0 | 0 | 0 | 11 | 1.6% |
| 20794791 (10 s) | walking on wet/slushy sidewalk | 21 | 0 | 0 | 0 | 0 | 21 | 0.0% |
| 6511420 (53 s) | snowy plaza / paved area | 107 | 89 | 0 | 18 | 0 | 0 | 21.6% |

**Speed (measured, MPS, 2 fps sampling, 1080x1920 input):** segmentation 633-708 ms per sampled frame, classification 25 ms per classified frame; wall time 0.41-0.48x real time (e.g. 53 s of video took 110 s). Output is written at the original frame rate.

## Inspection (visual, representative frames)
- **Clips 1 and 2:** the real sidewalks are segmented as *road* (pink covers only slivers). The pipeline therefore abstains on every frame (`no_sidewalk`). This is the intended safe behavior, but it means the pipeline produced no winter-condition output on two of three real sidewalk clips. We did not lower thresholds or invent sidewalk regions.
- **Clip 3:** the pink "sidewalk" covers snow-covered paved plaza/path surfaces (plausible, with holes where people stand and some ragged edges). Most frames are `snow` with scores near 1.0; 18 frames are `uncertain` (score ~0.7), where the mask changed shape. Visually the snow label looks reasonable, but this is not evidence of accuracy: there are no human labels, no `clear` frames exist in the footage, and the classifier was trained on roads and has never been validated on sidewalks.
- The crop fed to the classifier (see `crops/`) is a masked sidewalk region on grey, a different input style from its training images.

## What is NOT shown
- No accuracy: no human-labeled frames were used (the `--labels` comparison is implemented and unit-tested but has not been run on real labels).
- No `clear` sidewalk behavior was tested on real footage.
- No snow pixels, depth, ice or severity.

## Fastest way to make this meaningful
Label ~50-100 frames from clips with both clear and snowy sidewalks (`frame_index,label`), run with `--labels`, and collect pixel masks of snowy sidewalks to fine-tune the segmenter, which is the main failure point.
