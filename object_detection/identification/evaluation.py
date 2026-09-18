"""Measure how well a reference index identifies objects.

Two questions, which need different measurements:

1. **Known objects**: given a crop of an object that IS in the index, is the
   nearest reference the right one? That is top-1 accuracy.
2. **Unknown objects**: given a crop of an object that is NOT in the index,
   is its best score low enough to be rejected by the unknown threshold?
   Measured by leaving one object out of the index and querying with it.

The second matters because the architecture promises that new objects can be
added without retraining: the system must be able to say "I don't know this
one" instead of naming the closest reference.
"""

from dataclasses import dataclass

from PIL import Image

from object_detection.identification.index import ReferenceIndex


@dataclass(frozen=True)
class SeparationCounts:
    """How well known and unknown objects can be told apart by their scores.

    Both counts describe the best case for one side of the trade-off:
      - unknown_rejected: with the threshold set as low as possible while
        still keeping EVERY known crop, how many unknown crops are rejected?
      - known_kept: with the threshold set high enough to reject EVERY
        unknown crop, how many known crops still get identified?

    The counts are used instead of the raw score ranges because a single
    extreme crop moves a min or a max, but moves a count by exactly one.
    """

    unknown_rejected: int
    unknown_total: int
    known_kept: int
    known_total: int


def top1_accuracy(index: ReferenceIndex, labelled_crops) -> tuple[int, int]:
    """Return (correct, total): how often the nearest reference is the right object.

    `labelled_crops` is an iterable of (object name, crop), e.g. from
    cropping.iter_labelled_crops. Crops whose object is absent from the index
    can only ever be wrong, so query with crops of objects the index knows.
    """
    correct = 0
    total = 0
    for name, crop in labelled_crops:
        matches = index.search([crop], limit=1)[0]
        total += 1
        if matches and matches[0].object_id == name:
            correct += 1
    return correct, total


def best_scores(index: ReferenceIndex, labelled_crops, held_out: str) -> tuple[list[float], list[float]]:
    """Return (unknown_scores, known_scores) from an index built WITHOUT `held_out`.

    Each crop's best match score is collected, split by whether the crop
    shows the held-out object (unknown to this index) or another one (known).
    """
    unknown_scores = []
    known_scores = []
    for name, crop in labelled_crops:
        matches = index.search([crop], limit=1)[0]
        if not matches:
            continue
        (unknown_scores if name == held_out else known_scores).append(matches[0].score)
    return unknown_scores, known_scores


def separation(unknown_scores: list[float], known_scores: list[float]) -> SeparationCounts:
    """Count how cleanly the two sets of scores can be separated by a threshold."""
    if not unknown_scores or not known_scores:
        raise ValueError("both unknown and known scores are needed to measure separation")

    # Keep every known crop: the threshold can be no higher than the weakest
    # known score, and rejects unknown crops that score below it.
    rejected = sum(1 for score in unknown_scores if score < min(known_scores))
    # Reject every unknown crop: the threshold must clear the strongest
    # unknown score, and the known crops at or above it survive.
    kept = sum(1 for score in known_scores if score >= max(unknown_scores))
    return SeparationCounts(rejected, len(unknown_scores), kept, len(known_scores))


def fill_index(index: ReferenceIndex, labelled_crops, skip: str | None = None) -> int:
    """Store every crop as a reference under its object name; return how many.

    `skip` leaves one object out, which is how the unknown-object test builds
    an index that has never seen it.
    """
    stored = 0
    for name, crop in labelled_crops:
        if name == skip:
            continue
        index.add_references([crop], object_id=name, object_name=name)
        stored += 1
    return stored
