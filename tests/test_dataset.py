"""Tests for collapsing a multi-class export into the single-class dataset.

The test export is built in pytest's temporary folder, so nothing is written
near the real datasets.
"""

import yaml
from PIL import Image

from object_detection.utils.dataset import collapse_dataset, collapse_labels, collapse_split


def make_export(tmp_path, splits, path_prefix=""):
    """Write a small YOLO export: {split: {image name: label text or None}}."""
    data = {"nc": 3, "names": ["bag", "phone", "watch"]}
    for split, images in splits.items():
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        for name, label_text in images.items():
            Image.new("RGB", (20, 10)).save(tmp_path / split / "images" / f"{name}.jpg")
            if label_text is not None:
                (tmp_path / split / "labels" / f"{name}.txt").write_text(label_text)
        data[split] = f"{path_prefix}{split}/images"
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    return yaml_path


def test_class_ids_become_zero_and_coordinates_are_untouched():
    text = "2 0.5 0.5 0.2 0.2\n0 0.25 0.75 0.1 0.1\n"
    assert collapse_labels(text) == "0 0.5 0.5 0.2 0.2\n0 0.25 0.75 0.1 0.1"


def test_an_empty_label_file_stays_empty():
    # An empty label marks a background image: a deliberate negative example.
    assert collapse_labels("") == ""
    assert collapse_labels("\n\n") == ""


def test_every_split_in_the_export_is_collapsed(tmp_path):
    yaml_path = make_export(tmp_path, {
        "train": {"a": "1 0.5 0.5 0.2 0.2\n", "b": "2 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1\n"},
        "val": {"c": "2 0.5 0.5 0.2 0.2\n"},
        "test": {"d": "0 0.5 0.5 0.2 0.2\n"},
    })
    output = tmp_path / "collapsed"

    stats = collapse_dataset(yaml_path, output)

    # The test split is the point: the old notebook only handled train/valid.
    assert set(stats) == {"train", "val", "test"}
    assert (stats["train"].images, stats["train"].boxes) == (2, 3)
    assert (output / "test" / "images" / "d.jpg").exists()
    assert (output / "train" / "labels" / "b.txt").read_text() == "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1"


def test_background_images_are_kept(tmp_path):
    yaml_path = make_export(tmp_path, {
        "train": {"empty_label": "", "no_label": None, "labelled": "1 0.5 0.5 0.2 0.2\n"},
    })
    output = tmp_path / "collapsed"

    stats = collapse_dataset(yaml_path, output)

    # Both kinds of background (empty file, absent file) survive the copy.
    assert stats["train"].images == 3
    assert stats["train"].backgrounds == 2
    assert (output / "train" / "images" / "no_label.jpg").exists()
    assert (output / "train" / "labels" / "empty_label.txt").read_text() == ""


def test_the_written_data_yaml_is_portable(tmp_path):
    yaml_path = make_export(tmp_path, {"train": {"a": "1 0.5 0.5 0.2 0.2\n"}, "val": {"b": None}})
    output = tmp_path / "collapsed"

    collapse_dataset(yaml_path, output)
    written = yaml.safe_load((output / "data.yaml").read_text())

    # No absolute path: Ultralytics resolves these against the file's own
    # folder, so the dataset still works on another machine (e.g. Colab).
    assert "path" not in written
    assert written["train"] == "train/images"
    assert written["nc"] == 1
    assert written["names"] == ["object"]


def test_the_exports_folder_names_are_kept(tmp_path):
    # Roboflow names the validation folder "valid" while the data.yaml key is
    # "val"; the copy must keep the folder name, not the key.
    data = {"nc": 1, "names": ["bag"], "train": "train/images", "val": "valid/images"}
    for folder in ("train", "valid"):
        (tmp_path / folder / "images").mkdir(parents=True)
        (tmp_path / folder / "labels").mkdir(parents=True)
        Image.new("RGB", (20, 10)).save(tmp_path / folder / "images" / "a.jpg")
        (tmp_path / folder / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    output = tmp_path / "collapsed"

    collapse_dataset(yaml_path, output)

    assert (output / "valid" / "images" / "a.jpg").exists()
    assert yaml.safe_load((output / "data.yaml").read_text())["val"] == "valid/images"


def test_roboflow_style_parent_paths_are_found(tmp_path):
    # Roboflow writes "../train/images" relative to its data.yaml.
    yaml_path = make_export(tmp_path, {"train": {"a": "1 0.5 0.5 0.2 0.2\n"}}, path_prefix="../")
    output = tmp_path / "collapsed"

    stats = collapse_dataset(yaml_path, output)

    assert stats["train"].images == 1


def test_a_split_listed_but_missing_is_skipped(tmp_path):
    yaml_path = make_export(tmp_path, {"train": {"a": "1 0.5 0.5 0.2 0.2\n"}})
    # Claim a test split that was never exported.
    data = yaml.safe_load(yaml_path.read_text())
    data["test"] = "test/images"
    yaml_path.write_text(yaml.safe_dump(data))

    stats = collapse_dataset(yaml_path, tmp_path / "collapsed")

    assert set(stats) == {"train"}


def test_collapse_split_reports_what_it_copied(tmp_path):
    yaml_path = make_export(tmp_path, {"train": {"a": "1 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n", "b": ""}})
    stats = collapse_split(yaml_path.parent / "train" / "images", tmp_path / "out")

    assert (stats.images, stats.labels, stats.boxes, stats.backgrounds) == (2, 2, 2, 1)
