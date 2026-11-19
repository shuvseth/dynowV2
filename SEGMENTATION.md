# Phase 2: segmentation results and limitations

**Model:** `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` (SegFormer, ~27M params), pretrained by NVIDIA on Cityscapes. Used as-is; **no training or fine-tuning by the Dynow team.** Class ids/names come from the model config (`road`, `sidewalk`, ...; everything else = "other"). Input is resized so the long side is 1024 px (multiple of 32), logits are upsampled to the original resolution, and masks are the argmax class.

**Weights/data licensing:** Cityscapes is non-commercial research only; the HF model card lists license "other". Keep this in mind before any non-research use.

## What was inspected (2026-11-19, visually, 4 images, MPS)

| Image | Result |
|---|---|
| Ordinary summer street (Amsterdam, CC0, Wikimedia Commons) | Reasonable. Paved pedestrian areas mostly pink (sidewalk); red bike path and roadway purple (road); people, cars, trees, sky separated. Some sidewalk patches missed under people/shadows. |
| Stock video frame: icy/snowy footpath beside a road | **Failure.** The whole icy footpath is labeled *road* (76.7% road, 2.5% sidewalk). Confidence map is near-white (confident). No warning is raised. |
| Stock video frame: snowy plaza | **Mostly wrong/uncertain.** Snow-covered paved area labeled road; thin sidewalk strips at edges; mixed confidence along edges. No warning. |
| Stock video frame: wet/slushy sidewalk beside building | **Mostly wrong.** Walkable surface labeled road (41% road, 0.5% sidewalk); only a thin strip at the building edge is sidewalk. The low-confidence warning fired, but only because the tiny strip is uncertain. |

## Conclusions (limited by 4 images; no quantitative evaluation was done)

- The pipeline works end to end (masks, overlay, sidewalk-only image saved; directories supported; 28 tests pass).
- On clear streets the sidewalk mask looks plausible. **On snow/ice-covered sidewalks it cannot be trusted**: Cityscapes has almost no snow, and the model associates a snow-covered walkable surface with road. We did not assume bright pixels are sidewalk and did not repair masks.
- Model confidence is not a safety net: wrong snowy-sidewalk regions were high-confidence. A warning is emitted only when the sidewalk area is tiny or its mean confidence is low.
- An empty or small sidewalk mask is not evidence that no sidewalk exists.

## Optional next experiment (not started)

Fine-tune on winter sidewalk images with pixel masks. This needs polygon/mask annotations of snowy sidewalks (we have none; our 19 sidewalk photos are unlabeled). A small labeled set (~50-100 images with masks) would be the starting point; evaluate with per-class IoU on held-out images grouped by source.
