"""Tests for the evaluation helpers: IoU, label loading, and matching.

These are the functions that had real bugs while 02_evaluation was being
written (x_center used for y, max instead of min), so each of those mistakes
has a test that would catch it.
"""

import pytest
from PIL import Image

from object_detection.detection.evaluation import iou, load_true_boxes, match_detections
from object_detection.detection.inference import Detection


# --- iou ----------------------------------------------------------------------

@pytest.mark.parametrize(
    ("box_a", "box_b", "expected"),
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),     # identical
        ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),   # far apart: catches max/min mix-ups
        ((0, 0, 10, 10), (5, 0, 15, 10), 1 / 3),   # overlap 50, union 150
        ((0, 0, 10, 10), (10, 0, 20, 10), 0.0),    # edges touch, no shared area
        ((0, 0, 10, 10), (2, 2, 4, 4), 0.04),      # small box inside a big one: 4 / 100
    ],
)
def test_iou_known_values(box_a, box_b, expected):
    assert iou(box_a, box_b) == pytest.approx(expected)


def test_iou_is_symmetric():
    assert iou((0, 0, 10, 10), (5, 3, 15, 12)) == pytest.approx(iou((5, 3, 15, 12), (0, 0, 10, 10)))


def test_iou_of_zero_size_boxes_is_zero_not_a_crash():
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


# --- load_true_boxes ----------------------------------------------------------

def make_split(tmp_path, label_text, size=(200, 100)):
    """Create <tmp>/images/photo.jpg and, unless label_text is None, <tmp>/labels/photo.txt."""
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    img_path = tmp_path / "images" / "photo.jpg"
    Image.new("RGB", size).save(img_path)
    if label_text is not None:
        (tmp_path / "labels" / "photo.txt").write_text(label_text)
    return img_path


def test_load_true_boxes_converts_to_absolute_xyxy(tmp_path):
    # A non-square image (200 wide, 100 high) and a box whose x and y differ,
    # so swapping x_center/y_center or width/height gives a different answer.
    img_path = make_split(tmp_path, "0 0.25 0.75 0.1 0.2\n")
    # x_center 50, y_center 75, width 20, height 20 in pixels.
    assert load_true_boxes(img_path) == [pytest.approx((40, 65, 60, 85))]


def test_load_true_boxes_reads_every_line_and_skips_blank_ones(tmp_path):
    img_path = make_split(tmp_path, "0 0.5 0.5 1.0 1.0\n\n0 0.25 0.25 0.5 0.5\n")
    boxes = load_true_boxes(img_path)
    assert boxes == [pytest.approx((0, 0, 200, 100)), pytest.approx((0, 0, 100, 50))]


def test_empty_label_file_is_a_background_image(tmp_path):
    assert load_true_boxes(make_split(tmp_path, "")) == []


def test_missing_label_file_is_a_background_image(tmp_path):
    assert load_true_boxes(make_split(tmp_path, None)) == []


# --- match_detections ---------------------------------------------------------

TRUE_BOX = (0, 0, 10, 10)


def test_good_overlap_is_found():
    detection = Detection(bbox=(0, 0, 10, 9), confidence=0.9)
    assert match_detections([detection], [TRUE_BOX]) == ([detection], [], [])


def test_iou_of_exactly_the_threshold_counts_as_found():
    detection = Detection(bbox=(0, 0, 10, 5), confidence=0.9)  # overlap 50, union 100: IoU 0.5
    found, false_boxes, missed = match_detections([detection], [TRUE_BOX])
    assert (found, false_boxes, missed) == ([detection], [], [])


def test_poor_overlap_is_both_a_false_box_and_a_miss():
    detection = Detection(bbox=(5, 5, 15, 15), confidence=0.9)  # IoU 25 / 175
    assert match_detections([detection], [TRUE_BOX]) == ([], [detection], [TRUE_BOX])


def test_a_true_box_can_only_be_claimed_once():
    first = Detection(bbox=(0, 0, 10, 10), confidence=0.9)
    duplicate = Detection(bbox=(0, 0, 10, 9), confidence=0.6)
    assert match_detections([duplicate, first], [TRUE_BOX]) == ([first], [duplicate], [])


def test_most_confident_detection_claims_first_even_with_lower_iou():
    confident = Detection(bbox=(0, 0, 10, 6), confidence=0.9)   # IoU 0.6
    better_fit = Detection(bbox=(0, 0, 10, 10), confidence=0.5)  # IoU 1.0
    assert match_detections([better_fit, confident], [TRUE_BOX]) == ([confident], [better_fit], [])


def test_no_detections_means_every_true_box_is_missed():
    assert match_detections([], [TRUE_BOX, (20, 20, 30, 30)]) == ([], [], [TRUE_BOX, (20, 20, 30, 30)])


def test_on_a_background_image_every_detection_is_false():
    detection = Detection(bbox=(0, 0, 10, 10), confidence=0.9)
    assert match_detections([detection], []) == ([], [detection], [])


def test_matching_does_not_modify_the_callers_lists():
    detections = [Detection(bbox=(0, 0, 10, 10), confidence=0.9)]
    true_boxes = [TRUE_BOX]
    match_detections(detections, true_boxes)
    assert true_boxes == [TRUE_BOX]
    assert len(detections) == 1
