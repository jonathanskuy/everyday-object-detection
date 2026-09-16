"""Tests for LocalEmbeddingIndex, using a real Qdrant in local mode.

Qdrant's local mode stores its data in a folder, so each test gets its own
temporary one and no server is needed. The embedder is a stub that turns a
crop's colour into a 3-number vector: no model downloads, and similarity is
easy to reason about (a red crop is closest to the reddest reference).
"""

import pytest
from PIL import Image

from object_detection.identification.embedders import Embedder
from object_detection.identification.index import LocalEmbeddingIndex


class ColourEmbedder(Embedder):
    """Embeds a crop as its top-left pixel's RGB values, scaled to 0-1."""

    def __init__(self, name: str = "test/colour"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, crops: list[Image.Image]) -> list[list[float]]:
        return [[channel / 255 for channel in crop.convert("RGB").getpixel((0, 0))] for crop in crops]


def colour_crop(colour):
    return Image.new("RGB", (8, 8), colour)


RED = colour_crop((255, 0, 0))
GREEN = colour_crop((0, 255, 0))
BLUE = colour_crop((0, 0, 255))
REDDISH = colour_crop((200, 40, 40))


@pytest.fixture
def index(tmp_path):
    """A fresh, empty index in its own temporary folder, closed afterwards."""
    index = LocalEmbeddingIndex(ColourEmbedder(), str(tmp_path / "qdrant"), "test_objects")
    yield index
    index.close()


def test_search_finds_the_stored_object(index):
    index.add_references([RED], "obj-1", "red thing")

    matches = index.search([REDDISH], limit=1)

    assert len(matches) == 1           # one list of results per query crop
    assert len(matches[0]) == 1
    assert matches[0][0].object_id == "obj-1"
    assert matches[0][0].object_name == "red thing"
    assert 0 < matches[0][0].score <= 1


def test_best_match_comes_first_and_limit_is_respected(index):
    index.add_references([RED], "red", "red thing")
    index.add_references([GREEN], "green", "green thing")
    index.add_references([BLUE], "blue", "blue thing")

    matches = index.search([REDDISH], limit=2)[0]

    assert [match.object_id for match in matches][0] == "red"
    assert len(matches) == 2
    assert matches[0].score >= matches[1].score


def test_results_are_returned_in_the_order_of_the_query_crops(index):
    index.add_references([RED], "red", "red thing")
    index.add_references([BLUE], "blue", "blue thing")

    matches = index.search([BLUE, RED], limit=1)

    assert [result[0].object_id for result in matches] == ["blue", "red"]


def test_several_references_can_share_one_object(index):
    index.add_references([RED, REDDISH], "red", "red thing")
    index.add_references([BLUE], "blue", "blue thing")

    # Both red references are stored, so asking for three results returns
    # the same object twice plus the blue one.
    matches = index.search([RED], limit=3)[0]
    assert [match.object_id for match in matches] == ["red", "red", "blue"]


def test_remove_object_deletes_only_that_objects_references(index):
    index.add_references([RED, REDDISH], "red", "red thing")
    index.add_references([BLUE], "blue", "blue thing")

    index.remove_object("red")

    matches = index.search([RED], limit=5)[0]
    assert [match.object_id for match in matches] == ["blue"]


def test_empty_input_is_handled_without_calling_qdrant(index):
    index.add_references([], "red", "red thing")     # nothing to store
    assert index.search([], limit=1) == []


def test_reopening_with_a_different_embedder_is_refused(tmp_path):
    location = str(tmp_path / "qdrant")
    index = LocalEmbeddingIndex(ColourEmbedder("test/colour"), location, "test_objects")
    index.add_references([RED], "red", "red thing")
    index.close()

    # Vectors from another model aren't comparable with the stored ones, so
    # this must fail loudly instead of returning nonsense matches.
    with pytest.raises(ValueError, match="not comparable"):
        LocalEmbeddingIndex(ColourEmbedder("test/other-model"), location, "test_objects")


def test_reopening_with_the_same_embedder_keeps_the_references(tmp_path):
    location = str(tmp_path / "qdrant")
    first = LocalEmbeddingIndex(ColourEmbedder(), location, "test_objects")
    first.add_references([RED], "red", "red thing")
    first.close()

    reopened = LocalEmbeddingIndex(ColourEmbedder(), location, "test_objects")
    try:
        assert reopened.search([RED], limit=1)[0][0].object_id == "red"
    finally:
        reopened.close()
