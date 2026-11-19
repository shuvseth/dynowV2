# Training

## What has actually been trained

**One real run exists: `udot_proxy_v1` (below), a road-surface proxy model, not a sidewalk model.** The test suite trains a tiny head for 2-3 epochs on generated noise images purely to verify the code runs; those runs are discarded and say nothing about sidewalks. This file must be updated with real run details after a real run.

## Model design

- **Backbone:** DINOv2 **ViT-S/14** (`dinov2_vits14`, ~22M parameters, 384-d embedding), loaded from Meta's `facebookresearch/dinov2` via `torch.hub`. Pretrained by Meta AI with self-supervised learning on the curated LVD-142M image set. We did **not** train it.
- **Frozen:** every backbone parameter (`requires_grad=False`, kept in eval mode, run under `no_grad`).
- **Trained by us:** only the head: `Dropout(0.2) -> Linear(384, 2)` (set `head_hidden>0` for one hidden layer). 770 parameters by default.
- **Input:** images resized to 224x224 (must be a multiple of the 14-px patch), whole frame kept (no center crop), ImageNet mean/std normalization. Training augmentation: RandomResizedCrop (scale 0.6-1), horizontal flip, brightness/contrast/saturation jitter (no hue, no vertical flip).
- **Features:** the backbone's CLS embedding.
- **Loss:** cross-entropy on the 2 logits. AdamW (lr 1e-3, weight decay 0.01) over head parameters only.
- **Prediction:** argmax of the logits; the reported "model score" is the softmax value of that class. It is **not calibrated** (calibration has not been evaluated) and must not be read as a probability.
- **Checkpoint selection:** best validation accuracy (ties broken by lower validation loss). The test split is never loaded by `train`.

## Reproducibility

Each run directory `outputs/runs/<run_name>/` contains `config.yaml`, `environment.json` (torch version, device, git commit, manifest hash, parameter counts), `history.csv`, `best.pt`, `last.pt`. Checkpoints store only the head plus config, class names, versions and history; the backbone is re-fetched from torch.hub. Seeds are fixed (`seed`), but MPS/GPU kernels may not be bit-identical across runs. The hub backbone code tracks the dinov2 `main` branch; the environment file records the date implicitly via git commit and torch version only, so note the run date here when reporting results.

## Limitations

- The `clear`/`snow` model classifies the whole image, not just the sidewalk, and has no `uncertain` or ice output.
- Visible snow does not imply the sidewalk is inaccessible, and no severity is estimated.
- Small datasets give high-variance metrics; `evaluate` flags results `preliminary` below 100 test images or 30 per class. Results hold only for the conditions/locations in the data.

## Results

### Run `udot_proxy_v1` (2026-11-19): ROAD-SURFACE PROXY, not sidewalks

- **Data:** Roboflow Universe "UDOT WINTER ROAD CONDITIONS" v1 (windshield/dashcam road images; license CC BY 4.0 per the listing, verify on the dataset page). Only the Snow (3269) and Clear (579) folders were used; Wet and Slush dropped. 42 byte-identical duplicates removed. Labels are the dataset authors' (third-party), not ours, and were only spot-checked. Roboflow's own splits were ignored.
- **Splits:** re-split by approximate source group (68 groups guessed from filenames: screenshot date+hour, frame-number buckets; true video sources are unknown, so some leakage between near-identical frames may remain). train 434 clear / 1872 snow, val 88 / 670, test 57 / 727 (not touched yet).
- **Config:** `configs/udot_proxy.yaml` (10 epochs, batch 16, lr 1e-3, balanced class weights computed from train, seed 42, MPS). ~88 s/epoch.
- **Validation (what we have):** best checkpoint epoch 7, val accuracy 0.9763, val loss 0.0904. Validation is 88% snow, so always predicting snow scores ~88%; per-class recall has not been inspected yet. Val accuracy moved between 0.953 and 0.976 across epochs, which is noise-level for this size.
- **Test results:** none. The held-out test split and the 19-image sidewalk folder have deliberately NOT been used. Do not quote a test number until `evaluate` has been run once.
- **Caveat:** the model was trained on dashcam road views. It has not been shown to work on sidewalk photos.

### Run `udot_plus10_v1` (2026-11-19): UDOT roads + 10 snowy sidewalk photos

- **Data:** same as `udot_proxy_v1` plus 10 web-collected snowy sidewalk photos (all labeled `snow`, forced into train, repeated x10 per epoch via `repeat` in `configs/udot_plus10.yaml`). 3 of the 19 photos were excluded from the pool (two stock-photo watermarks, one identifiable person). The other 9 photos have never been used for training or evaluation. Their license is UNKNOWN: internal experiment only, do not redistribute.
- **Validation (UDOT only):** best val accuracy 0.9815 at epoch 4 (val loss 0.0853); epochs ranged 0.934-0.982. Validation contains no sidewalk images, so it says nothing about sidewalks.
- **Limits:** the 10 sidewalk images are all `snow`, so no clear sidewalks were seen; the model may learn "sidewalk-looking = snow". No test result exists for this run.
