"""Draw detections onto images for visual inspection in notebooks."""

from PIL import Image, ImageDraw, ImageFont

from object_detection.detection.inference import Detection

BOX_COLOUR = (255, 56, 56)
TEXT_COLOUR = (255, 255, 255)


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Return a copy of `image` with each detection's box and confidence drawn on it.

    The input image is left untouched, so the same image can be drawn on
    again (e.g. at a different confidence threshold).
    """
    # convert() always returns a new image, and forcing RGB also handles
    # greyscale or RGBA inputs that would otherwise reject a colour outline.
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Scale line and text size with the image: a fixed 2px line is invisible
    # on a 1600px photo once the notebook shrinks it to a thumbnail.
    longest_side = max(annotated.size)
    line_width = max(2, round(longest_side / 400))
    font = ImageFont.load_default(size=max(12, round(longest_side / 50)))

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOUR, width=line_width)

        # Label sits on a filled strip just above the box's top-left corner,
        # or just inside the box when the box touches the top of the image.
        label = f"{detection.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_height = bottom - top
        label_y = y1 - text_height - 2 * line_width
        if label_y < 0:
            label_y = y1
        draw.rectangle(
            (x1, label_y, x1 + (right - left) + 2 * line_width, label_y + text_height + 2 * line_width),
            fill=BOX_COLOUR,
        )
        draw.text((x1 + line_width, label_y + line_width - top), label, fill=TEXT_COLOUR, font=font)

    return annotated
