# Fine-tuning the sidewalk segmenter (human-drawn masks)

Why: pretrained segmenters (Cityscapes, ADE20K) mislabel snowy sidewalks: they paint snow verges as sidewalk and the
real walkway as road. Downloaded photos have no masks, so the masks must be drawn by a person.

## Data
`data/raw/seg_images/` (git-ignored) holds 38 Wikimedia Commons photos picked by hand because a walkable path is
visible (licences and authors in `PROVENANCE.csv`: CC0 / CC BY / CC BY-SA, attribution required if published).
Eleven `yardseries_*` photos come from one photographer/yard and share a split group. Add your own phone photos
of real routes to the same folder: they are the most valuable.

## Label (about 30-60 s per photo; AI-assisted outlining is built in)
```bash
labelme data/raw/seg_images --autosave --nodata --labels sidewalk,no_sidewalk
```
Draw a polygon (or use the "AI polygon" tool and click the path) around the WALKABLE path surface, whatever covers
it. Do not include the snow verge beside it. Label every visible sidewalk in the photo; everything unlabelled
becomes "other". Photos with no sidewalk: add a tiny `no_sidewalk` shape. Unlabelled photos are skipped.

## Prepare, train, evaluate
```bash
python -m dynow_vision.seg_data prepare --source "Wikimedia Commons + own photos" --license "mixed CC, see PROVENANCE.csv"
python -m dynow_vision.seg_train train --run-name seg_v1
python -m dynow_vision.seg_train eval --model outputs/seg_runs/seg_v1/best --split test   # once, at the end
```
`train` refuses with fewer than 20 train / 5 val labelled photos. It prints sidewalk IoU for the fine-tuned model and
for the untouched pretrained ADE20K model on the same validation photos. Only adopt the fine-tuned model if it wins
on held-out photos, and report the numbers with the small-sample caveat. To use it:
`python -m dynow_vision.analyze_image --image photo.jpg --checkpoint ... --seg-model outputs/seg_runs/seg_v1/best`.
