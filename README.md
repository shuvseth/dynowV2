# dynowV2 (Dynow Vision)

Computer vision research for detecting snow and evaluating winter sidewalk accessibility using semantic segmentation and DINOv2.

**Phase 1 scope:** a two-class (`clear` vs `snow`) image classifier = frozen pretrained DINOv2 ViT-S/14 + a classification head trained by us.
Phase 2 adds road/sidewalk segmentation with a pretrained model (below). Phases 3-4 add a video pipeline and human-assigned severity. Not in scope: ice detection, snow depth, automated severity prediction, any UI.

## Status (read this first)

| Item | State |
|---|---|
| Environment, MPS check | working (see below) |
| Dataset tooling, labeling CLI, splits | implemented, unit-tested |
| Training / evaluation / inference code | implemented, tested end-to-end on **synthetic** images (mechanics only) |
| **Real labeled images** | **none yet** |
| **Trained checkpoint / real metrics** | **none yet. No model has been trained on real data.** |

Phase 1 is not complete until real labeled images exist and `train` + `evaluate` have been run on them.

## Install

Requires Python 3.10+ (tested on 3.11, Apple Silicon, PyTorch 2.x with MPS).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
pytest -q
```

The first model load downloads DINOv2 code + weights (~85 MB) into `~/.cache/torch/hub`. CPU is used automatically if MPS is unavailable (`--device cpu` forces it).

## Add images and label them

1. Register images (copies them into `data/raw/` and records provenance; **source and license are mandatory**):

```bash
# frames from ONE video: give them a shared group so they never straddle train/test
python -m dynow_vision.dataset add path/to/frames --source "own video 2026-11-17 Michigan Ave" --license "own work" --group video_michave_01

# or derive the group from the file name: video_a_0001.jpg -> group "video_a"
python -m dynow_vision.dataset add path/to/frames --source "..." --license "..." --group-regex '^(.+)_\d+$'
```
Images already sorted into class folders (e.g. a Roboflow "Folder Structure" export) can take their labels from the folder names. These are third-party labels, so spot-check them; unmapped folders (e.g. `Wet`) are skipped:

```bash
python -m dynow_vision.dataset add ~/Downloads/roboflow_snow --source "Roboflow UDOT winter road conditions" --license "CC BY 4.0 (verify on dataset page)" --label-map Snow=snow Clear=clear
```

Without `--group`/`--group-regex`, each image is its own group, which is only correct if images are truly independent photos.

2. Label (opens each image in Preview; keys: `c` clear, `s` snow, `u` uncertain, Enter skip, `b` back, `q` quit; saved after every answer):

```bash
python -m dynow_vision.label            # unlabeled images only
python -m dynow_vision.label --relabel  # revisit everything
```
Only label images where the sidewalk/path is visible. Use `u` when unsure; uncertain images are excluded from training.

3. Check and split (group-aware, seeded). **Re-run `split` after any label or image change** (labeling clears splits):

```bash
python -m dynow_vision.dataset validate
python -m dynow_vision.dataset split --seed 42
python -m dynow_vision.dataset summary
```

Annotations live in `data/annotations/manifest.csv` (`filename,source,license,group,label,split`); see [DATASET.md](DATASET.md).

## Train / evaluate / predict

```bash
python -m dynow_vision.train --config configs/phase1.yaml --run-name first_run
python -m dynow_vision.train --config configs/phase1.yaml --resume outputs/runs/first_run/last.pt   # continue
python -m dynow_vision.evaluate --checkpoint outputs/runs/first_run/best.pt                          # held-out test split
python -m dynow_vision.predict --image some.jpg --checkpoint outputs/runs/first_run/best.pt          # writes outputs/predictions/some.json
```

Training refuses to start if any split has fewer images per class than `min_per_class` in the config. Details: [TRAINING.md](TRAINING.md).

## Phase 2: road / sidewalk segmentation (pretrained, not trained by us)

Uses NVIDIA's SegFormer-B2 fine-tuned on Cityscapes via Hugging Face `transformers` (downloaded once, ~110 MB, runs on MPS or CPU). We do **not** train or fine-tune it. Cityscapes is licensed for non-commercial research only.

```bash
pip install -r requirements.txt           # adds transformers
python -m dynow_vision.segment --input photo.jpg                       # one image
python -m dynow_vision.segment --input some_folder/ --out outputs/segmentation/my_run   # directory
```

Per image it writes `outputs/segmentation/<name>/`: `original.png`, `overlay.png` (Cityscapes colours: road purple, sidewalk pink), `road_mask.png`, `sidewalk_mask.png` (0/255), `sidewalk_only.png` (sidewalk pixels, rest black), `confidence.png`, `summary.json` (area fractions, confidence, warnings).

**Known limitation, read before using:** on snow/ice-covered sidewalks the model often labels the sidewalk as *road* and is confident about it, so `confidence.png` does not flag it. Details and inspected examples: [SEGMENTATION.md](SEGMENTATION.md).

## Phase 3: video pipeline (segmentation -> sidewalk crop -> classifier)

```bash
python -m dynow_vision.pipeline --video clip.mp4 --checkpoint outputs/runs/udot_proxy_v1/best.pt
# useful options: --sample-fps 2  --max-frames 300  --crop-mode masked|box  --min-sidewalk-fraction 0.03
#                 --min-seg-confidence 0.6  --min-clf-score 0.7  --labels my_labels.csv
```

Output in `outputs/video/<video name>/`: `annotated.mp4` (H.264, plays in QuickTime), `predictions.json` (per-frame status, scores, timestamps, per-stage ms, model identifiers, parameters, limitations), `masks/` (road + sidewalk PNGs per sampled frame), `crops/` (exact images fed to the classifier), `representative/` (one annotated frame per outcome).

Per sampled frame the outcome is one of: `snow`, `clear`, `uncertain` (classifier score below threshold), `unreliable_sidewalk` (segmentation confidence too low), `no_sidewalk` (sidewalk below 3% of the frame). Only a reliable sidewalk is ever classified. In the default `masked` crop mode every non-sidewalk pixel is replaced by a neutral grey, so snow on the road cannot influence the sidewalk label. Labels are per crop, not snow pixels. Frames between samples reuse the latest result (shown as "held").

Compare with your own labels: write a CSV `frame_index,label` (`clear`/`snow`, frame indices from `predictions.json`) and pass `--labels`. Results report coverage and accuracy with abstentions shown separately. See [PHASE3.md](PHASE3.md) for measured results and limitations.

## Phase 4: human-assigned severity (no automated severity model)

A 1-5 obstruction rubric plus "not assessable" (see [SEVERITY.md](SEVERITY.md)). Severity is rated by humans and shown next to the model outputs; it is a preliminary research rubric, not an accessibility certification or a snow-depth measurement.

```bash
python -m dynow_vision.severity frames --video clip.mp4
python -m dynow_vision.severity annotate --annotator yourname --images data/raw/severity_frames/clip
python -m dynow_vision.severity agreement      # needs >=2 annotators on the same frames
python -m dynow_vision.severity status
python -m dynow_vision.pipeline --video clip.mp4 --checkpoint outputs/runs/udot_proxy_v1/best.pt   # picks up data/annotations/severity/
```

## Layout

`src/dynow_vision/` code · `configs/` run configs · `data/` (images are git-ignored; manifest is tracked) · `outputs/` run artifacts (git-ignored) · `tests/`
