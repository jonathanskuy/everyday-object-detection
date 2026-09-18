"""Cut detections out of an image, ready to be embedded.

Cropping decides WHAT the embedding model sees, and that is the same for
every model: which pixels belong to the detection, how much surrounding area
to include, and making the crop square. It deliberately does NOT resize or
normalise: each embedding model needs its own input size and colour
normalisation, and FastEmbed and Qdrant Cloud Inference do that preparation
themselves. Resizing here would resize twice and bias the embedding model
comparison. See embedders.py for the model-specific part.

Reference crops (used to build the index) and query crops (from the detector
at runtime) MUST both come from this module with the same settings.
Otherwise their vectors aren't comparable, and matching degrades without any
error.
"""

import math
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from object_detection.utils.labels import load_labelled_boxes

# Fill colour for the borders added when padding a crop to a square. A neutral
# mid-grey (the same one Ultralytics uses when letterboxing) is less likely to
# be mistaken for part of an object than black or white.
PAD_COLOUR = (114, 114, 114)


def crop_detection(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    padding: float,
    min_size: int,
) -> Image.Image | None:
    """Return the square crop for one detection, or None if the box is too small to use.

    Args:
        image: the upright image the detector ran on, i.e. after EXIF
            orientation was applied, so `bbox` refers to the right pixels.
        bbox: (x1, y1, x2, y2) in absolute pixels, as in Detection.bbox.
        padding: extra area on each side, as a fraction of the box's width
            and height (0.1 adds 10% per side). Will come from the config.
        min_size: boxes narrower or shorter than this many pixels (after
            clamping) return None; a few pixels carry no usable information.
            Will come from the config.

    Steps, in order:
        1. Expand the box by `padding` on each side.
        2. Round outward to whole pixels and clamp to the image edges.
        3. Return None if the result is smaller than `min_size`.
        4. Cut the region out of the image.
        5. Pad it to a square with PAD_COLOUR, keeping the object centred.
           This preserves the object's true shape (stretching would distort
           it by an amount that depends on the box, not the product), and
           means an embedding model's own centre-crop can't cut off the ends
           of long objects.

    The returned image is RGB and has any size; the embedder resizes it for
    its model. The input image is never modified.
    """
    x1, y1, x2, y2 = bbox

    # 1. Padding is relative to the box, so a small object and a large one
    # get the same proportion of surrounding context.
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding

    # 2. Round outward (floor the left/top edge, ceil the right/bottom edge):
    # a pixel the box only partly covers is kept rather than cut off. Then
    # clamp, since padding (or a box at the border) can reach past the image.
    left = max(0, math.floor(x1 - pad_x))
    top = max(0, math.floor(y1 - pad_y))
    right = min(image.width, math.ceil(x2 + pad_x))
    bottom = min(image.height, math.ceil(y2 + pad_y))

    # 3. Also catches boxes entirely outside the image, where clamping makes
    # right <= left and the width zero or negative.
    width = right - left
    height = bottom - top
    if width < min_size or height < min_size:
        return None

    # 4. crop() returns a new image; converting to RGB means every crop has
    # the same colour format whatever the source image was (greyscale, RGBA).
    region = image.crop((left, top, right, bottom)).convert("RGB")

    # 5. A square as large as the longer side, filled with grey, with the
    # region pasted in the middle. `//` is integer division: when the
    # leftover space is odd, the extra pixel of grey goes on the right/bottom.
    side = max(width, height)
    square = Image.new("RGB", (side, side), PAD_COLOUR)
    square.paste(region, ((side - width) // 2, (side - height) // 2))
    return square


def iter_labelled_crops(
    img_paths: list[Path],
    class_names: dict[int, str],
    padding: float,
    min_size: int,
) -> Iterator[tuple[str, Image.Image]]:
    """Yield (object name, crop) for every labelled box in `img_paths`.

    Used both to build the reference index and to evaluate it, so references
    and evaluation crops are always produced the same way.

    Yields one crop at a time instead of returning a list: a dataset of tens
    of thousands of images would not fit in memory all at once. Boxes too
    small to crop are skipped, and background images yield nothing, which is
    correct here — they contain no object to reference.
    """
    for img_path in img_paths:
        with Image.open(img_path) as image:
            for class_id, bbox in load_labelled_boxes(img_path):
                crop = crop_detection(image, bbox, padding=padding, min_size=min_size)
                if crop is not None:
                    yield class_names[class_id], crop
