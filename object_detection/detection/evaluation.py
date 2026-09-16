"""Compare detections against YOLO-format ground-truth labels.

Usage:
    from object_detection.detection.evaluation import load_true_boxes, match_detections

    true_boxes = load_true_boxes(img_path)
    found, false_boxes, missed = match_detections(detector.predict(img_path), true_boxes)

Extracted unchanged from notebooks/02_evaluation.ipynb, with one necessary
difference: the notebook found label files through its VALID variable, which
a package module can't see. load_true_boxes now uses YOLO's folder convention
instead (see its docstring), which gives the same paths for the dataset.
"""

from object_detection.utils.labels import load_labelled_boxes

# A prediction counts as finding a labelled object when their IoU is at least
# this. 0.5 is part of the definition of the metric (the "50" in mAP50), not a
# tunable setting like the confidence threshold, so it stays out of the config.
IOU_THRESHOLD = 0.5


def load_true_boxes(img_path):
    """Return one image's labelled boxes as absolute-pixel (x1, y1, x2, y2) tuples.

    The label file is found by YOLO's convention, the same one Ultralytics
    uses: an image at <split>/images/<name>.<ext> has its labels at
    <split>/labels/<name>.txt. A missing or empty label file means a
    background image, and returns an empty list.

    The detector is class-agnostic, so the class of each label is dropped
    here; utils.labels does the reading and the conversion to absolute pixels.
    """

    return [box for _, box in load_labelled_boxes(img_path)]


def iou(box_a, box_b):
    """Return the Intersection over Union of two absolute-pixel (x1, y1, x2, y2) boxes."""

    overlap_x1 = max(box_a[0], box_b[0])
    overlap_y1 = max(box_a[1], box_b[1])
    overlap_x2 = min(box_a[2], box_b[2])
    overlap_y2 = min(box_a[3], box_b[3])
    overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - overlap_area
    if union_area == 0:
        return 0.0
    return overlap_area / union_area


def match_detections(detections, true_boxes):
    """Split one image's results into found detections, false detections, and missed true boxes.

    Matching is one-to-one, and the most confident detection claims first: a
    second box on an already-claimed object counts as a false detection.
    """

    found = []
    false_boxes = []
    unmatched = list(true_boxes)

    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        best_iou = 0.0
        best_box = None
        for true_box in unmatched:
            overlap = iou(detection.bbox, true_box)
            if overlap > best_iou:
                best_iou = overlap
                best_box = true_box
        if best_box is not None and best_iou >= IOU_THRESHOLD:
            found.append(detection)
            unmatched.remove(best_box)
        else:
            false_boxes.append(detection)
    return found, false_boxes, unmatched
