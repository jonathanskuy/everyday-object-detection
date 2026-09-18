"""Tests for the identification metrics and for producing labelled crops."""

import pytest
from PIL import Image

from object_detection.identification.cropping import iter_labelled_crops
from object_detection.identification.evaluation import (
    best_scores,
    fill_index,
    separation,
    top1_accuracy,
)
from object_detection.identification.index import Match, ReferenceIndex


class FakeIndex(ReferenceIndex):
    """Answers every search with the match queued for that crop, and records what was stored."""

    def __init__(self, matches=()):
        self._matches = list(matches)
        self.stored = []

    def search(self, crops, limit):
        return [[self._matches.pop(0)] if self._matches else [] for _ in crops]

    def add_references(self, crops, object_id, object_name):
        self.stored.extend((object_id, crop) for crop in crops)

    def remove_object(self, object_id):
        raise AssertionError("not used in these tests")


CROP = Image.new("RGB", (8, 8))


def test_top1_accuracy_counts_correct_nearest_matches():
    index = FakeIndex([Match("watch", "watch", 0.9), Match("phone", "phone", 0.8), Match("bag", "bag", 0.7)])

    correct, total = top1_accuracy(index, [("watch", CROP), ("glasses", CROP), ("bag", CROP)])

    assert (correct, total) == (2, 3)


def test_a_crop_with_no_match_at_all_counts_as_wrong():
    correct, total = top1_accuracy(FakeIndex(), [("watch", CROP)])
    assert (correct, total) == (0, 1)


def test_best_scores_splits_the_held_out_object_from_the_rest():
    index = FakeIndex([Match("perfume", "perfume", 0.4), Match("phone", "phone", 0.9)])

    unknown, known = best_scores(index, [("watch", CROP), ("phone", CROP)], held_out="watch")

    assert unknown == [0.4]
    assert known == [0.9]


def test_separation_counts_a_clean_split():
    # Every unknown score is below every known score.
    counts = separation(unknown_scores=[0.2, 0.3], known_scores=[0.7, 0.8])

    assert (counts.unknown_rejected, counts.unknown_total) == (2, 2)
    assert (counts.known_kept, counts.known_total) == (2, 2)


def test_separation_counts_an_overlap():
    # 0.75 sits inside the known range: no threshold separates them fully.
    counts = separation(unknown_scores=[0.2, 0.75], known_scores=[0.7, 0.9])

    # Keeping both known crops (threshold 0.7) rejects only the 0.2 crop.
    assert counts.unknown_rejected == 1
    # Rejecting both unknown crops (threshold above 0.75) keeps only 0.9.
    assert counts.known_kept == 1


def test_separation_needs_both_sides():
    with pytest.raises(ValueError):
        separation(unknown_scores=[], known_scores=[0.5])


def test_fill_index_stores_every_crop_under_its_name():
    index = FakeIndex()

    stored = fill_index(index, [("watch", CROP), ("phone", CROP)])

    assert stored == 2
    assert [name for name, _ in index.stored] == ["watch", "phone"]


def test_fill_index_can_leave_one_object_out():
    index = FakeIndex()

    stored = fill_index(index, [("watch", CROP), ("phone", CROP)], skip="watch")

    assert stored == 1
    assert [name for name, _ in index.stored] == ["phone"]


def test_iter_labelled_crops_names_each_crop_and_skips_backgrounds(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for name in ("one", "two"):
        Image.new("RGB", (200, 100)).save(tmp_path / "images" / f"{name}.jpg")
    (tmp_path / "labels" / "one.txt").write_text("0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.2 0.2\n")
    (tmp_path / "labels" / "two.txt").write_text("")   # background image

    crops = list(iter_labelled_crops(
        sorted((tmp_path / "images").glob("*")),
        {0: "bag", 1: "watch"},
        padding=0,
        min_size=1,
    ))

    assert [name for name, _ in crops] == ["bag", "watch"]
    assert all(crop.width == crop.height for _, crop in crops)   # squared for the embedder


def test_iter_labelled_crops_skips_boxes_that_are_too_small(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "images" / "one.jpg")
    # 2 px wide box, plus a usable one.
    (tmp_path / "labels" / "one.txt").write_text("0 0.5 0.5 0.01 0.5\n1 0.25 0.25 0.2 0.2\n")

    crops = list(iter_labelled_crops(
        sorted((tmp_path / "images").glob("*")), {0: "bag", 1: "watch"}, padding=0, min_size=16
    ))

    assert [name for name, _ in crops] == ["watch"]
