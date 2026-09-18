"""Tests for train(): it must pass the config's settings to Ultralytics.

Training itself is never run here; a stand-in YOLO records the arguments.
"""

import pytest

import object_detection.detection.train as train_module
from object_detection.config.loader import PROJECT_ROOT, load_config


class RecordingYOLO:
    """Stands in for ultralytics.YOLO and records what train() passes to it."""

    last = None

    def __init__(self, weights):
        self.weights = weights
        self.train_kwargs = None
        RecordingYOLO.last = self

    def train(self, **kwargs):
        self.train_kwargs = kwargs


def test_train_passes_config_values_to_ultralytics(monkeypatch):
    monkeypatch.setattr(train_module, "YOLO", RecordingYOLO)
    cfg = load_config()

    train_module.train(cfg)

    model = RecordingYOLO.last
    assert model.weights == cfg.model.weights
    assert model.train_kwargs == {
        "data": str(cfg.data.detector_yaml),
        "epochs": cfg.train.epochs,
        "batch": cfg.train.batch,
        "imgsz": cfg.train.imgsz,
        "seed": cfg.train.seed,
        "project": str(PROJECT_ROOT / "runs"),
        "name": cfg.train.name,
    }


def test_resuming_continues_from_the_runs_last_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(train_module, "YOLO", RecordingYOLO)
    monkeypatch.setattr(train_module, "PROJECT_ROOT", tmp_path)
    cfg = load_config()
    last = tmp_path / "runs" / cfg.train.name / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.touch()

    train_module.train(cfg, resume=True)

    model = RecordingYOLO.last
    assert model.weights == last
    # Only resume=True: the settings come from the interrupted run itself.
    assert model.train_kwargs == {"resume": True}


def test_resuming_without_a_checkpoint_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(train_module, "YOLO", RecordingYOLO)
    monkeypatch.setattr(train_module, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="cannot resume"):
        train_module.train(load_config(), resume=True)
