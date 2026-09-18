"""Draw detections onto images for visual inspection in notebooks."""

from PIL import Image, ImageDraw, ImageFont

from object_detection.detection.inference import Detection
from object_detection.identification.identify import Identification

BOX_COLOUR = (255, 56, 56)
# Identification: green for an object the index recognised, red for one it
# did not, so a glance at the picture says which is which.
IDENTIFIED_COLOUR = (0, 180, 60)
UNKNOWN_COLOUR = BOX_COLOUR
TEXT_COLOUR = (255, 255, 255)


def _prepare(image: Image.Image) -> tuple[Image.Image, ImageDraw.ImageDraw, int, ImageFont.FreeTypeFont]:
    """Return a drawable copy of `image` plus sizes that scale with it.

    convert() always returns a new image, so the caller's image is never
    modified, and forcing RGB also handles greyscale or transparent inputs
    that would otherwise reject a colour outline.

    A fixed 2px line and 12pt text are invisible on a 1600px photo once a
    notebook shrinks it to a thumbnail, so both scale with the longest side.
    """
    annotated = image.convert("RGB")
    longest_side = max(annotated.size)
    return (
        annotated,
        ImageDraw.Draw(annotated),
        max(2, round(longest_side / 400)),
        ImageFont.load_default(size=max(12, round(longest_side / 50))),
    )


def _draw_labelled_box(draw, bbox, label: str, colour, line_width: int, font) -> None:
    """Draw one box with a filled label strip at its top-left corner."""
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2, y2), outline=colour, width=line_width)

    # The strip sits just above the box, or just inside it when the box
    # touches the top of the image and there is no room above.
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_height = bottom - top
    label_y = y1 - text_height - 2 * line_width
    if label_y < 0:
        label_y = y1
    draw.rectangle(
        (x1, label_y, x1 + (right - left) + 2 * line_width, label_y + text_height + 2 * line_width),
        fill=colour,
    )
    draw.text((x1 + line_width, label_y + line_width - top), label, fill=TEXT_COLOUR, font=font)


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Return a copy of `image` with each detection's box and confidence drawn on it.

    The input image is left untouched, so the same image can be drawn on
    again (e.g. at a different confidence threshold).
    """
    annotated, draw, line_width, font = _prepare(image)
    for detection in detections:
        _draw_labelled_box(draw, detection.bbox, f"{detection.confidence:.2f}", BOX_COLOUR, line_width, font)
    return annotated


def draw_identifications(image: Image.Image, identifications: list[Identification]) -> Image.Image:
    """Return a copy of `image` showing what each detection was identified as.

    Green boxes are objects found in the reference index, labelled with the
    object's name and how well it matched. Red boxes are detections the index
    did not recognise: they are still objects, and still counted, but nothing
    similar enough is registered.
    """
    annotated, draw, line_width, font = _prepare(image)
    for identification in identifications:
        if identification.object_name is None:
            label, colour = "unknown", UNKNOWN_COLOUR
        else:
            label, colour = f"{identification.object_name} {identification.match_score:.2f}", IDENTIFIED_COLOUR
        _draw_labelled_box(draw, identification.detection.bbox, label, colour, line_width, font)
    return annotated
