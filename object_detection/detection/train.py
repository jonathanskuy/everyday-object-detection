"""Fine-tune the pretrained YOLO checkpoint into the single-class detector.

Usage:
    from object_detection.config.loader import load_config
    from object_detection.detection.train import train

    results = train(load_config())
    results = train(load_config(), resume=True)   # after an interrupted run

Extracted from notebooks/01_training.ipynb, so a notebook and any future
script run exactly the same training call.
"""

from ultralytics import YOLO

from object_detection.config.loader import PROJECT_ROOT, Config


def train(cfg: Config, resume: bool = False):
    """Train the detector with the settings in `cfg` and return Ultralytics' metrics.

    Weights, plots and results.csv are written to runs/<train.name>/.
    Ultralytics never overwrites an existing run folder: if it already exists,
    it writes to runs/<train.name>-2/, -3/, ... instead. Either way,
    `inference.weights` in the config has to be pointed at the new run.

    With `resume=True`, training continues from runs/<train.name>/weights/
    last.pt, which Ultralytics saves after every epoch. Use it when a long run
    was interrupted, e.g. a Colab session that disconnected. The settings then
    come from the interrupted run itself, not from `cfg`: resuming with
    different settings would produce a run that is neither.
    """
    if resume:
        last = PROJECT_ROOT / "runs" / cfg.train.name / "weights" / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(
                f"cannot resume: {last} does not exist. Check train.name in the config "
                f"matches the interrupted run."
            )
        return YOLO(last).train(resume=True)

    model = YOLO(cfg.model.weights)
    return model.train(
        data=str(cfg.data.detector_yaml),
        epochs=cfg.train.epochs,
        batch=cfg.train.batch,
        imgsz=cfg.train.imgsz,
        seed=cfg.train.seed,
        project=str(PROJECT_ROOT / "runs"),
        name=cfg.train.name,
    )
