"""Tests for reading YOLO label files.

Stage 1 drops the class, Stage 2 needs it, so both go through
utils.labels.load_labelled_boxes and the class id is tested here.
"""

from PIL import Image

from object_detection.utils.labels import load_labelled_boxes


def make_split(tmp_path, label_text, size=(200, 100)):
    """Create <tmp>/images/photo.jpg and, unless label_text is None, <tmp>/labels/photo.txt."""
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    img_path = tmp_path / "images" / "photo.jpg"
    Image.new("RGB", size).save(img_path)
    if label_text is not None:
        (tmp_path / "labels" / "photo.txt").write_text(label_text)
    return img_path


def test_class_id_and_absolute_box_are_returned(tmp_path):
    # Non-square image, and a box whose x and y differ, so an x/y mix-up shows.
    img_path = make_split(tmp_path, "3 0.25 0.75 0.1 0.2\n")
    (class_id, box), = load_labelled_boxes(img_path)
    assert class_id == 3
    assert box == (40, 65, 60, 85)


def test_every_line_is_read_and_blank_lines_skipped(tmp_path):
    img_path = make_split(tmp_path, "0 0.5 0.5 1.0 1.0\n\n7 0.25 0.25 0.5 0.5\n")
    assert [class_id for class_id, _ in load_labelled_boxes(img_path)] == [0, 7]


def test_empty_label_file_is_a_background_image(tmp_path):
    assert load_labelled_boxes(make_split(tmp_path, "")) == []


def test_missing_label_file_is_a_background_image(tmp_path):
    assert load_labelled_boxes(make_split(tmp_path, None)) == []
