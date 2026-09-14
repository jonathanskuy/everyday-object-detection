"""Fine-tune the pretrained YOLO checkpoint into the single-class detector.

Usage:
    from object_detection.config.loader import load_config
    from object_detection.detection.train import train

    results = train(load_config())

Moved unchanged from notebooks/01_training.ipynb, so a notebook and any
future script run exactly the same training call.
"""

from ultralytics import YOLO

from object_detection.config.loader import PROJECT_ROOT, Config


def train(cfg: Config):
    """Train the detector with the settings in `cfg` and return Ultralytics' metrics.

    Weights, plots and results.csv are written to runs/baseline/. Ultralytics
    never overwrites an existing run folder: if runs/baseline/ already exists,
    it writes to runs/baseline-2/, runs/baseline-3/, ... instead, and
    `inference.weights` in the config has to be pointed at the new one.
    """
    model = YOLO(cfg.model.weights)
    return model.train(
        data=str(cfg.data.detector_yaml),
        epochs=cfg.train.epochs,
        batch=cfg.train.batch,
        imgsz=cfg.train.imgsz,
        seed=cfg.train.seed,
        project=str(PROJECT_ROOT / "runs"),
        # Hardcoded, as it was in the notebook. A candidate for the YAML
        # config, since every retrain currently produces a "baseline-<N>".
        name="baseline",
    )
