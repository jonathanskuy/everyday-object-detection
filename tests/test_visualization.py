"""Tests for draw_detections."""

from PIL import Image

from object_detection.detection.inference import Detection
from object_detection.utils.visualization import draw_detections

DETECTIONS = [
    Detection(bbox=(20, 30, 120, 90), confidence=0.91),
    Detection(bbox=(0, 0, 40, 40), confidence=0.5),  # touches the top edge: label must fit inside
]


def test_returns_an_annotated_copy_of_the_same_size():
    image = Image.new("RGB", (200, 150), "white")
    annotated = draw_detections(image, DETECTIONS)
    assert annotated is not image
    assert annotated.size == image.size
    assert annotated.tobytes() != image.tobytes()


def test_leaves_the_input_image_untouched():
    image = Image.new("RGB", (200, 150), "white")
    before = image.tobytes()
    draw_detections(image, DETECTIONS)
    assert image.tobytes() == before


def test_accepts_non_rgb_images():
    for mode in ("L", "RGBA"):
        annotated = draw_detections(Image.new(mode, (200, 150)), DETECTIONS)
        assert annotated.mode == "RGB"


def test_no_detections_gives_an_unchanged_copy():
    image = Image.new("RGB", (200, 150), "white")
    assert draw_detections(image, []).tobytes() == image.tobytes()
