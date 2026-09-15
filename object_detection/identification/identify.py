"""Identify which object each detection is: crop, search the index, apply the unknown threshold.

This is where Stage 2 fills in the three fields Stage 1 reserves in the API
response (object_id, object_name, match_score). It changes their values
only; the response shape stays exactly the same.

Counting is not affected: the number of products in an image is the number of
detections, decided by Stage 1 alone. Detections that can't be identified are
still detections.

Status: scaffold only. Nothing is implemented yet.
"""

from dataclasses import dataclass

from PIL import Image

from object_detection.detection.inference import Detection
from object_detection.identification.index import ReferenceIndex


@dataclass(frozen=True)
class Identification:
    """A detection together with what Stage 2 concluded about it.

    For an unknown item, all three identity fields are None (null in the
    API), rather than naming the closest wrong object.
    """

    detection: Detection
    object_id: str | None
    object_name: str | None
    match_score: float | None


def identify(
    image: Image.Image,
    detections: list[Detection],
    index: ReferenceIndex,
    padding: float,
    min_size: int,
    unknown_threshold: float,
) -> list[Identification]:
    """Return one Identification per detection, in the same order as `detections`.

    Args:
        image: the upright image the detector ran on (after EXIF orientation).
        detections: the detector's output for that image.
        index: any ReferenceIndex; this function doesn't know or care whether
            FastEmbed, DINOv2 or Qdrant Cloud Inference is behind it.
        padding, min_size: passed to cropping.crop_detection. The index must
            have been built with the same values.
        unknown_threshold: minimum similarity for a match to count. Below it,
            the item is reported as unknown. Will come from the config, and
            has to be chosen by measurement, like the detector's `conf`.

    Steps:
        1. Crop every detection with cropping.crop_detection.
        2. Search the index for all usable crops in one batch, taking only
           the single best match for each.
        3. For each detection: if its best match's score is at least
           `unknown_threshold`, copy that match's object_id, object_name and
           score; otherwise (score too low, or the box was too small to
           crop) leave all three as None.
    """
    raise NotImplementedError
