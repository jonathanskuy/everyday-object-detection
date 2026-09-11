# Everyday Object Detection

Proof-of-concept system that recognises everyday objects (bags, glasses,
phones, watches, ...) in images. It finds where each object is, then works
out which object it is.

**Status:** Stage 1 (detection) is in progress. Stage 2 (identification) has
not started.

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
fills them in without changing the shape of the response. Bounding boxes are
`[x1, y1, x2, y2]` in absolute pixels.

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
├── api/                 FastAPI app (not built yet)
├── detection/           Stage 1: YOLO training and inference
├── identification/      Stage 2: embedding + retrieval (not started)
├── config/              YAML config (default.yaml) and its loader
└── utils/               shared helpers
scripts/                 standalone command-line tools
notebooks/               experiments; import from object_detection
```

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

## Checking a dataset

Before training on a dataset export, check it:

```bash
python scripts/check_dataset.py path/to/data.yaml
```

The script reports image, background and box counts per split, and flags
label files that Ultralytics would reject. It works with any number of
classes.

## Dataset

93 images, 238 annotated boxes across 8 object classes, collapsed to a single item class for detection. 15 background images with no annotations, included deliberately as negatives. Split 75/18 train/validation.