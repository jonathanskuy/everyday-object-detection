"""Tests for identify(): cropping, searching and the unknown threshold.

A fake index returns fixed matches and records what it was asked, so these
tests need no embedding model and no Qdrant.
"""

from PIL import Image

from object_detection.detection.inference import Detection
from object_detection.identification.identify import identify
from object_detection.identification.index import Match, ReferenceIndex

IMAGE = Image.new("RGB", (200, 200), "white")
BIG = Detection(bbox=(10, 10, 90, 90), confidence=0.9)
ALSO_BIG = Detection(bbox=(100, 100, 180, 180), confidence=0.8)
TINY = Detection(bbox=(10, 10, 14, 14), confidence=0.7)  # 4 px: below min_size


class FakeIndex(ReferenceIndex):
    """Returns the queued matches in order, one per searched crop."""

    def __init__(self, matches_per_crop):
        self._matches_per_crop = list(matches_per_crop)
        self.searched_crops = []

    def search(self, crops, limit):
        self.searched_crops.extend(crops)
        return [self._matches_per_crop.pop(0) for _ in crops]

    def add_references(self, crops, object_id, object_name):
        raise AssertionError("identify() must not write to the index")

    def remove_object(self, object_id):
        raise AssertionError("identify() must not write to the index")


def test_a_confident_match_fills_in_the_identity():
    index = FakeIndex([[Match("watch-1", "watch", 0.82)]])

    result = identify(IMAGE, [BIG], index, padding=0.1, min_size=16, unknown_threshold=0.5)

    assert len(result) == 1
    assert result[0].detection is BIG
    assert (result[0].object_id, result[0].object_name, result[0].match_score) == ("watch-1", "watch", 0.82)


def test_a_weak_match_is_reported_as_unknown():
    index = FakeIndex([[Match("watch-1", "watch", 0.30)]])

    result = identify(IMAGE, [BIG], index, padding=0.1, min_size=16, unknown_threshold=0.5)

    # Naming the closest reference anyway would be a confident wrong answer.
    assert (result[0].object_id, result[0].object_name, result[0].match_score) == (None, None, None)
    assert result[0].detection is BIG


def test_a_score_exactly_at_the_threshold_is_accepted():
    index = FakeIndex([[Match("watch-1", "watch", 0.5)]])

    result = identify(IMAGE, [BIG], index, padding=0.1, min_size=16, unknown_threshold=0.5)

    assert result[0].object_id == "watch-1"


def test_an_empty_index_leaves_everything_unknown():
    index = FakeIndex([[]])

    result = identify(IMAGE, [BIG], index, padding=0.1, min_size=16, unknown_threshold=0.5)

    assert result[0].object_id is None


def test_results_follow_the_order_of_the_detections():
    index = FakeIndex([[Match("a", "first", 0.9)], [Match("b", "second", 0.9)]])

    result = identify(IMAGE, [BIG, ALSO_BIG], index, padding=0.1, min_size=16, unknown_threshold=0.5)

    assert [identification.object_id for identification in result] == ["a", "b"]
    assert [identification.detection for identification in result] == [BIG, ALSO_BIG]


def test_a_box_too_small_to_crop_stays_in_the_list_as_unknown():
    # Only the two usable crops are searched, but all three detections are
    # returned: the number of detections is the product count, and must not
    # change because one box was unusable.
    index = FakeIndex([[Match("a", "first", 0.9)], [Match("b", "second", 0.9)]])

    result = identify(IMAGE, [BIG, TINY, ALSO_BIG], index, padding=0.1, min_size=16, unknown_threshold=0.5)

    assert len(index.searched_crops) == 2
    assert [identification.object_id for identification in result] == ["a", None, "b"]


def test_no_detections_means_no_identifications():
    index = FakeIndex([])

    assert identify(IMAGE, [], index, padding=0.1, min_size=16, unknown_threshold=0.5) == []


def test_crops_are_square_and_padded_as_configured():
    index = FakeIndex([[Match("a", "first", 0.9)], [Match("b", "second", 0.9)]])

    identify(IMAGE, [ALSO_BIG, BIG], index, padding=0.25, min_size=16, unknown_threshold=0.5)

    padded, clamped = index.searched_crops
    # ALSO_BIG is an 80 px box with room around it: 80 + 25% on each side = 120.
    assert padded.size == (120, 120)
    # BIG sits 10 px from the top-left corner, so its padding is cut off
    # there: 10 + 80 + 20 = 110, and the crop is still square.
    assert clamped.size == (110, 110)
