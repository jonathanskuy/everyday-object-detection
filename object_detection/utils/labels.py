"""Read YOLO-format label files.

A label file has one line per object, "class x_center y_center width height",
with the four coordinates normalised to 0-1. Everything downstream works in
absolute-pixel (x1, y1, x2, y2), the same format as Detection.bbox, so the
conversion happens here, once.

Detection only needs the boxes (it collapses every class into one); Stage 2
needs the class as well, because the class name is the object's identity in
the reference index. Both read labels through this module.
"""

from pathlib import Path

from PIL import Image


def load_labelled_boxes(img_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Return (class_id, (x1, y1, x2, y2)) for every labelled box in one image.

    The label file is found by YOLO's convention, the same one Ultralytics
    uses: an image at <split>/images/<name>.<ext> has its labels at
    <split>/labels/<name>.txt. A missing or empty label file means a
    background image, and returns an empty list.
    """
    label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    if not label_path.exists():
        return []

    with Image.open(img_path) as image:
        img_width, img_height = image.size

    labelled_boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = line.split()
        x1 = (float(x_center) - float(width) / 2) * img_width
        y1 = (float(y_center) - float(height) / 2) * img_height
        x2 = (float(x_center) + float(width) / 2) * img_width
        y2 = (float(y_center) + float(height) / 2) * img_height
        labelled_boxes.append((int(class_id), (x1, y1, x2, y2)))

    return labelled_boxes
