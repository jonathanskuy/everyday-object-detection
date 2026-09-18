"""Tests for drawing detections and identifications."""

from PIL import Image

from object_detection.detection.inference import Detection
from object_detection.identification.identify import Identification
from object_detection.utils.visualization import (
    IDENTIFIED_COLOUR,
    UNKNOWN_COLOUR,
    draw_detections,
    draw_identifications,
)

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


IDENTIFIED = Identification(
    detection=Detection(bbox=(20, 30, 120, 90), confidence=0.91),
    object_id="watch",
    object_name="watch",
    match_score=0.84,
)
UNKNOWN = Identification(
    detection=Detection(bbox=(130, 30, 190, 90), confidence=0.6),
    object_id=None,
    object_name=None,
    match_score=None,
)


def colours_used(annotated):
    return {colour for _, colour in annotated.getcolors(maxcolors=100000)}


def test_identified_objects_are_drawn_in_the_identified_colour():
    annotated = draw_identifications(Image.new("RGB", (300, 200), "white"), [IDENTIFIED])

    assert IDENTIFIED_COLOUR in colours_used(annotated)
    assert UNKNOWN_COLOUR not in colours_used(annotated)


def test_unidentified_detections_are_still_drawn_in_the_unknown_colour():
    # They are objects, and still counted; they just have no name.
    annotated = draw_identifications(Image.new("RGB", (300, 200), "white"), [UNKNOWN])

    assert UNKNOWN_COLOUR in colours_used(annotated)
    assert IDENTIFIED_COLOUR not in colours_used(annotated)


def test_both_kinds_can_appear_in_one_image():
    annotated = draw_identifications(Image.new("RGB", (300, 200), "white"), [IDENTIFIED, UNKNOWN])

    assert {IDENTIFIED_COLOUR, UNKNOWN_COLOUR} <= colours_used(annotated)


def test_drawing_identifications_leaves_the_input_image_untouched():
    image = Image.new("RGB", (300, 200), "white")
    before = image.tobytes()
    draw_identifications(image, [IDENTIFIED, UNKNOWN])
    assert image.tobytes() == before
