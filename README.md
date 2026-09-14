# Everyday Object Detection

Proof-of-concept system that recognises everyday objects (bags, glasses,
phones, watches, ...) in images. It finds where each object is, then works
out which object it is.

**Status:** Stage 1 (detection) is complete: the detector is trained,
evaluated, and served through an HTTP API. Stage 2 (identification) has not
started.

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

Each detection carries identity fields that Stage 1 leaves as `null`. Stage 2
fills them in without changing the shape of the response.

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
├── identification/      Stage 2: embedding + retrieval (not started)
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
`object` class in a separate copy (`notebooks/00_data_preparation.ipynb`); the
original export is kept untouched, because its per-class labels will seed the
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

1. **`00_data_preparation`**: collapses the 8-class export into the
   single-class detection dataset.
2. **`01_training`**: fine-tunes the pretrained YOLO checkpoint (settings from
   the config) and inspects predictions.
3. **`02_evaluation`**: measures the trained detector and chooses the
   confidence threshold.

Outputs go to `runs/<name>/`, where the name is `train.name` in the config.
Before retraining, give the experiment a new `train.name` (Ultralytics never
overwrites a run; reusing a name produces `<name>-2`). Afterwards, point
`inference.weights` at the new run's `best.pt`, and re-run `02_evaluation`,
which names its output `<name>-val` after the run it evaluates.

## Results

Baseline detector, evaluated in `notebooks/02_evaluation.ipynb` on the
validation split:

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

**Caveats:** there is no separate test split yet. The validation set was used
both to select the best training epoch and to choose the threshold, so these
numbers are optimistic, and the validation set is small.

**Known weaknesses**, which point to data rather than settings:
- **Overlapping objects** get merged into one box spanning several objects,
  causing both false boxes and misses.
- **Unusual poses**, such as a bag leaning against a wall, get low confidence.
- **Labelling consistency** needs a rule, e.g. whether straps belong inside
  the box, and whether everyday objects outside the 8 types are labelled too.

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