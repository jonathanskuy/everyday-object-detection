# Everyday Object Detection

Proof-of-concept system that recognises everyday objects (bags, glasses,
phones, watches, ...) in images. It finds where each object is, then works
out which object it is.

**Status:** both stages work end to end. The detector is trained and
evaluated, reference objects are indexed in Qdrant, and the HTTP API returns
detected objects with their identification.

## How it works

The system has two stages, and keeping them separate is deliberate.

1. **Detection:** a YOLO detector finds *where* objects are in the image. It
   is trained on a single class, `object`, so it has no idea *what* each
   object is. It only draws boxes.
2. **Identification:** each detected box is cropped, turned into an embedding
   vector, and looked up in a vector index of known objects. The closest
   match says *which* object it is.

### Why the detector is class-agnostic

The set of objects we want to recognise keeps changing: new types of object,
new instances of existing ones, objects whose appearance changes. If the
detector classified objects itself, every change would mean relabelling the
dataset and retraining the model.

With a single-class detector plus vector lookup, the detector only needs to
be good at finding "an object". Recognising a new one means adding a few
vectors to the index, with no retraining and no redeploy.

### Response format

The number of entries in `detections` is the number of products found in the
image. Every detection is returned, whether or not it could be identified:
`object_id`, `object_name` and `match_score` describe the match, and are
`null` when no reference object was similar enough.

Bounding boxes are `[x1, y1, x2, y2]`: the top-left and bottom-right corners
in **absolute pixels**, not normalised 0-1 values. They refer to the image as
it is normally displayed (after its EXIF orientation is applied), whose size
is `image_width` x `image_height`.

```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.91,
      "object_id": null,
      "object_name": null,
      "match_score": null
    }
  ],
  "image_width": 1920,
  "image_height": 1080
}
```

## Repository layout

```
object_detection/        the installable package (all application code)
├── api/                 FastAPI app: endpoint, response schemas, model loading
├── detection/           Stage 1: YOLO training and inference
├── identification/      Stage 2: embedding + retrieval (scaffold)
├── config/              YAML config (default.yaml) and its loader
└── utils/               shared helpers (drawing detections)
scripts/                 standalone command-line tools
notebooks/               data preparation, training and evaluation; import from object_detection
tests/                   automated tests (pytest)
```

Created locally and not committed: `datasets/` (the Roboflow export and its
single-class copy), `runs/` (training and evaluation outputs, including the
trained weights) and `weights/` (the pretrained checkpoint, downloaded
automatically on first training run).

Notebooks only drive the work. The training and inference code they validate
lives in `object_detection/detection/`, and the API calls that same code, so
what the notebooks measured is what the service runs.

Settings such as model size, epochs, batch size, image size, dataset paths and
the confidence threshold live in
[`object_detection/config/default.yaml`](object_detection/config/default.yaml),
not in code.

## Setup

Requires Python 3.10+. Commands are run from the repository root.

1. Create and activate an environment:

   ```bash
   conda create -n everyday-od python=3.11
   conda activate everyday-od
   ```

2. **If you have an NVIDIA GPU**, install a CUDA build of PyTorch first. The
   default PyTorch that pip installs on Windows is CPU-only, and training on
   CPU is very slow. Choose the command for your CUDA version at
   <https://pytorch.org/get-started/locally/>, for example:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   python -c "import torch; print(torch.cuda.is_available())"   # should print True
   ```

3. Install the package in editable mode, with the notebook tools:

   ```bash
   pip install -e ".[dev]"
   ```

   Editable (`-e`) means Python imports the code straight from this folder,
   so edits take effect without reinstalling. Leave out `[dev]` if you don't
   need Jupyter.

4. If you'll commit notebooks, enable output stripping once per clone:

   ```bash
   nbstripout --install
   ```

   This strips cell outputs from notebooks as they are committed, keeping
   diffs readable and images out of git history. Your local copy keeps its
   outputs.

## Running the tests

```bash
pytest
```

The tests cover the config loader, the conversion of Ultralytics' output into
`Detection` objects, the evaluation helpers (IoU, label loading, matching),
the arguments `train()` passes to Ultralytics, drawing, and the `/detect`
endpoint (response shape, EXIF rotation, error codes). They use fake models
and images generated on the fly, so they run in seconds and never train
anything. One test runs the real detector on a validation image; it is
skipped automatically on machines without the trained weights and dataset.

## Checking a dataset

Before training on a dataset export, check it:

```bash
python scripts/check_dataset.py path/to/data.yaml
```

The script reports image, background and box counts per split, and flags
label files that Ultralytics would reject. It works with any number of
classes.

## Dataset

A Roboflow export in YOLOv8 format (`train/`, `valid/`, `data.yaml`) with 8
object classes. For detection, the classes are collapsed into a single
`object` class in a separate copy (`scripts/collapse_dataset.py`); the
original export is kept untouched, because its per-class labels seed the
Stage 2 reference index.

Images with no annotations are deliberate background negatives: they teach
the detector not to fire on empty surfaces, and are never filtered out.

The dataset is still growing, so counts aren't listed here. For the current
numbers, run the check script on either export:

```bash
python scripts/check_dataset.py datasets/collapsed/data.yaml
```

## Workflow

The notebooks run in order, and each builds on the previous one:

1. **`scripts/collapse_dataset.py`**: copies the export into the single-class
   dataset the detector trains on (`notebooks/00_data_preparation.ipynb`
   inspects the export around it).

   ```bash
   python scripts/collapse_dataset.py               # build it
   python scripts/collapse_dataset.py --overwrite   # rebuild from scratch
   ```

   Splits and classes come from the export's own `data.yaml`, so a test split
   is collapsed too, and the copy's `data.yaml` holds no absolute path: it
   works on another machine, e.g. in Colab.
2. **`01_training`**: fine-tunes the pretrained YOLO checkpoint (settings from
   the config) and inspects predictions.
3. **`02_evaluation`**: measures the trained detector and chooses the
   confidence threshold.
4. **`scripts/build_reference_index.py`**: fills the reference index from the
   per-class export (see below).
5. **`03_identification_evaluation`**: measures identification end to end and
   chooses the unknown threshold.

Outputs go to `runs/<name>/`, where the name is `train.name` in the config.
Before retraining, give the experiment a new `train.name` (Ultralytics never
overwrites a run; reusing a name produces `<name>-2`). Afterwards, point
`inference.weights` at the new run's `best.pt`, and re-run `02_evaluation`,
which names its output `<name>-val` after the run it evaluates.

## Results

All numbers below come from the validation split, through the same code the
API runs.

### Detection

Baseline detector, evaluated in `notebooks/02_evaluation.ipynb`:

| Metric | Value |
|---|---|
| mAP50 | 0.973 |
| mAP50-95 | 0.903 |

The confidence threshold is set to **0.5** (Ultralytics' default is 0.25). It
was chosen by counting found, false and missed boxes at thresholds from 0.1 to
0.9, through the same `Detector` code the API uses. 0.5 removes most false
boxes while staying well below the threshold where real objects start to be
dropped. Missed objects are treated as the more costly error, since Stage 2
can reject a false box but can never recover a missed object. The full
reasoning is in the notebook.

At that threshold the detector finds 41 of the 45 labelled objects and draws
5 boxes that are not objects.

### Identification

Evaluated in `notebooks/03_identification_evaluation.ipynb`, against an index
built from the train split's crops:

| Metric | Value |
|---|---|
| Top-1 accuracy, on detected objects | 40/41 |
| Unknown objects rejected at threshold 0.5 | 5/7 |
| Known objects lost at threshold 0.5 | 0/41 |

`unknown_threshold` is set to **0.5**. Above it an object is named, below it
the detection is returned with null identification fields. It was chosen by
holding one object out of the index entirely, so its crops stand in for an
object that has never been registered: at 0.5 most of those are rejected
while no known object is lost. Raising it to 0.6 rejects all of them, but
costs three correct names.

The single identification error is a bag with headphones lying on it: the
crop contains both objects, and the headphones dominate it.

### Choice of embedding model

`scripts/compare_embedders.py` measures candidates the same way, holding each
class out in turn (45 unknown and 315 known crops in total):

| Embedder | Dim | Top-1 | AUC | Unknown rejected |
|---|---|---|---|---|
| **Qdrant/Unicom-ViT-B-16** (in use) | 768 | 0.98 | 0.983 | 41/45 |
| Qdrant/clip-ViT-B-32-vision | 512 | 0.98 | 0.964 | 17/45 |
| Qdrant/resnet50-onnx | 2048 | 1.00 | 0.986 | 36/45 |
| facebook/dinov2-small@224 | 384 | 0.96 | 0.990 | 38/45 |
| facebook/dinov2-base@224 | 768 | 0.96 | 0.991 | 37/45 |

AUC is how often a known crop outscores an unknown one, over every pair; it
is reported alongside the counts because a count can be decided by a single
extreme crop. CLIP is clearly weakest at recognising that an object is not in
the index, which fits a model trained to match captions rather than
individual objects. The rest are within one or two crops of each other, so
Unicom is kept and the comparison should be repeated on a larger dataset.

### Caveats

- **No separate test split yet.** The validation set selected the training
  epoch and both thresholds, so these numbers are optimistic.
- **The validation set is small**: one crop is worth about 2 percentage
  points.
- **Each class is a single physical item**, so identification here means
  recognising a type. Telling apart several similar items of the same type is
  harder and untested.

### Known weaknesses

These point to data rather than settings:
- **Overlapping objects** get merged into one box spanning several objects,
  causing false boxes, misses and wrong counts — and, when two objects share
  a crop, wrong identification too.
- **Unusual poses**, such as a bag leaning against a wall, get low confidence.
- **Labelling consistency** needs a rule, e.g. whether straps belong inside
  the box, and whether everyday objects outside the 8 types are labelled too.

## Building the reference index

Identification matches each detected crop against a reference index. Build it
from the original per-class export, whose labels already carry object names:

```bash
python scripts/build_reference_index.py           # index the train split
python scripts/build_reference_index.py --reset   # replace what is stored
```

Only the train split is indexed, so the validation split's crops can measure
identification accuracy on objects the index has not seen. The embedding
model is downloaded on first use.

The index lives where `identification.qdrant_location` points. A folder means
Qdrant's local mode, which **one process at a time** can open: stop the API
before rebuilding the index, or move Qdrant to a server (see the config).

## Comparing embedding models

The embedding model decides how well objects are told apart, and whether an
unregistered object can be recognised as unknown. To compare candidates on
the current dataset:

```bash
python scripts/compare_embedders.py --held-out all
python scripts/compare_embedders.py --model dinov2:facebook/dinov2-base:518
```

Each candidate is measured on the same crops: identification accuracy against
an index of everything, then separation against an index with one object left
out. Temporary indexes are used, so the project's own index is untouched.
Models are downloaded on first use, and `--held-out all` repeats the test for
every class, which takes a while but stops one object deciding the result.

## Running the API

Start the server from the repository root:

```bash
uvicorn object_detection.api.app:app
```

The model loads once at startup; wait for `Application startup complete`.
Then open <http://127.0.0.1:8000/docs> for interactive documentation, where
you can upload an image to `POST /detect` straight from the browser.

From the command line:

```bash
curl -F "file=@path/to/photo.jpg" http://127.0.0.1:8000/detect
```

- The response follows the [response format](#response-format) above.
- Photos with EXIF rotation (common from phones) are rotated upright before
  detection, so boxes match the image as displayed.
- A file that isn't a readable image returns **400**; a request without a
  file returns **422**.
- The model weights and confidence threshold come from the `inference`
  section of the config, which is read at startup. Restart the server after
  changing it.

To use the detector from Python instead:

```python
from object_detection.config.loader import load_config
from object_detection.detection.inference import Detector

cfg = load_config()
detector = Detector(cfg.inference.weights, cfg.inference.conf)
for detection in detector.predict("path/to/photo.jpg"):
    print(f"{detection.bbox} {detection.confidence:.3f}")
```