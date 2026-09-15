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

Status: scaffold only. Nothing is implemented yet.
"""

from PIL import Image

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
        2. Round to whole pixels and clamp to the image edges.
        3. Return None if the result is smaller than `min_size`.
        4. Cut the region out of the image.
        5. Pad it to a square with PAD_COLOUR, keeping the object centred.
           This preserves the object's true shape (stretching would distort
           it by an amount that depends on the box, not the product), and
           means an embedding model's own centre-crop can't cut off the ends
           of long objects.

    The returned image has any size; the embedder resizes it for its model.
    """
    raise NotImplementedError
