# Dataset

**Current contents: 3848 UDOT road images (579 clear, 3269 snow), imported with third-party labels. No sidewalk images yet.**

## Manifest: `data/annotations/manifest.csv`

| column | meaning |
|---|---|
| filename | path relative to `data/raw/` |
| source | where it came from (URL, dataset name, "own photo/video ...") |
| license | license / usage terms (required for labeled images) |
| group | original video or photo session; all rows of a group go to the same split |
| label | `clear`, `snow`, `uncertain`, or empty (unlabeled) |
| split | `train` / `val` / `test`, assigned by `dataset split`; empty for uncertain/unlabeled |

## Label definitions (Phase 1)

- **clear**: the sidewalk/pedestrian path is visible and shows no snow cover.
- **snow**: the sidewalk/pedestrian path is visible with snow on it.
- **uncertain**: path not visible, ambiguous, dark/blurry, or you cannot decide. Excluded from training.

Ice is not a class in Phase 1 (transparent ice generally cannot be judged from RGB).

## Splitting rules

- Seeded (`--seed`), default 70/15/15 by image count, stratified per class at the group level.
- Whole groups move together, so near-duplicate frames from one video never appear in both train and test. `validate` and training both fail on leakage.
- Test split is only read by `evaluate`. Choose hyperparameters from validation results only.
- With few groups the split is coarse; with fewer than 100 test images or 30 per class, `evaluate` marks results **preliminary**.

## Sources and licenses

Record every source here when images are added (the manifest holds per-image detail):

| source | license | images | notes |
|---|---|---|---|
| Roboflow Universe, UDOT WINTER ROAD CONDITIONS v1 | CC BY 4.0 per listing (verify) | 3848 (Snow+Clear) | dashcam road views; groups approximated from filenames |

Images are not committed to git (`data/raw/` is ignored). Only add images you have the right to use.
