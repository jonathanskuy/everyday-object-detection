"""Tests for Detector: turning Ultralytics' output into plain Detection objects."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import object_detection.detection.inference as inference
from object_detection.config.loader import load_config
from object_detection.detection.inference import Detection, Detector


class FakeYOLO:
    """Stands in for ultralytics.YOLO: returns fixed boxes and records how it was called.

    Lets the conversion logic be tested without model weights, a GPU, or the
    time it takes to load a real model.
    """

    def __init__(self, weights):
        self.weights = weights
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=torch.tensor([[10.0, 20.0, 110.0, 220.0], [5.0, 6.0, 7.0, 8.0]]),
            conf=torch.tensor([0.9, 0.3]),
        )
        return [SimpleNamespace(boxes=boxes)]


@pytest.fixture
def fake_detector(monkeypatch):
    # monkeypatch swaps YOLO inside the inference module for this test only,
    # and puts the real one back afterwards.
    monkeypatch.setattr(inference, "YOLO", FakeYOLO)
    return Detector("some/weights.pt", conf=0.25)


def test_predict_returns_plain_detections(fake_detector):
    detections = fake_detector.predict(Image.new("RGB", (300, 300)))
    assert detections == [
        Detection(bbox=(10.0, 20.0, 110.0, 220.0), confidence=pytest.approx(0.9)),
        Detection(bbox=(5.0, 6.0, 7.0, 8.0), confidence=pytest.approx(0.3)),
    ]
    assert all(isinstance(detection.bbox, tuple) for detection in detections)


def test_predict_passes_the_configured_threshold(fake_detector):
    fake_detector.predict(Image.new("RGB", (300, 300)))
    assert fake_detector._model.calls[0]["conf"] == 0.25


def test_detections_are_read_only():
    detection = Detection(bbox=(0, 0, 1, 1), confidence=0.5)
    with pytest.raises(AttributeError):
        detection.confidence = 0.9


# --- Real model, if available -------------------------------------------------
# Weights and the dataset are not committed, so this test only runs on a
# machine that has both; elsewhere pytest reports it as skipped.

cfg = load_config()
VALID_IMAGES = sorted((cfg.data.detector_yaml.parent / "valid" / "images").glob("*"))


@pytest.mark.skipif(
    not cfg.inference.weights.exists() or not VALID_IMAGES,
    reason="needs the trained weights and the collapsed dataset",
)
def test_real_detector_returns_sensible_boxes():
    detector = Detector(cfg.inference.weights, cfg.inference.conf)
    img_path = VALID_IMAGES[0]
    with Image.open(img_path) as image:
        width, height = image.size

    detections = detector.predict(img_path)

    assert detections, "expected at least one detection on a validation image with objects"
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert cfg.inference.conf <= detection.confidence <= 1
