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

from dataclasses import dataclass

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


@dataclass(frozen=True)
class CountSummary:
    """How well the number of detections matches the number of labelled objects.

    The count of products in an image is simply how many detections it has,
    so counting is decided by the detector alone: identification never adds
    or removes one.

    WARNING: a correct count does not mean correct detections. A false box
    and a missed object in the same image cancel out, so an image with two
    mistakes can still count perfectly. Always report this beside precision
    and recall, never instead of them.
    """

    images: int
    exact: int          # images whose count is exactly right
    overcounted: int    # detections too many, summed over images
    undercounted: int   # objects missed in the count, summed over images

    @property
    def exact_share(self) -> float:
        """Share of images counted exactly right, from 0 to 1."""
        return self.exact / self.images if self.images else 0.0

    @property
    def mean_absolute_error(self) -> float:
        """Average size of the per-image count error, ignoring its direction."""
        return (self.overcounted + self.undercounted) / self.images if self.images else 0.0

    @property
    def net_error(self) -> int:
        """Total signed error: positive means the system counts too many overall.

        Near zero does not mean accurate: an image counting +3 and another
        counting -3 cancel out. Compare it with mean_absolute_error to see
        whether errors mostly lean one way.
        """
        return self.overcounted - self.undercounted


def count_summary(counted_and_true) -> CountSummary:
    """Summarise counting over a dataset.

    `counted_and_true` is an iterable of (detections in the image, labelled
    objects in the image) pairs, one per image — including images with no
    objects, where drawing any box is an overcount.
    """
    images = exact = overcounted = undercounted = 0
    for counted, true_count in counted_and_true:
        images += 1
        error = counted - true_count
        if error == 0:
            exact += 1
        elif error > 0:
            overcounted += error
        else:
            undercounted += -error
    return CountSummary(images, exact, overcounted, undercounted)


def pair_detections_with_boxes(detections, true_boxes):
    """Pair each detection with the true box it found, and list the boxes nobody found.

    Returns (pairs, missed):
      - pairs: one (detection, true_box) per detection, in order of
        decreasing confidence. true_box is None for a detection that found
        nothing, i.e. a false detection.
      - missed: the true boxes no detection claimed.

    Matching is one-to-one, and the most confident detection claims first: a
    second box on an already-claimed object counts as a false detection.

    Keeping the pairing (rather than only the counts) lets a caller look up
    whatever else it knows about a true box — Stage 2 needs its class, to
    check whether the detection was identified as the right object.
    """

    pairs = []
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
            pairs.append((detection, best_box))
            unmatched.remove(best_box)
        else:
            pairs.append((detection, None))
    return pairs, unmatched


def match_detections(detections, true_boxes):
    """Split one image's results into found detections, false detections, and missed true boxes.

    A counting-oriented view of pair_detections_with_boxes: a detection that
    found a box is "found", one that found nothing is a false detection.
    """

    pairs, missed = pair_detections_with_boxes(detections, true_boxes)
    found = [detection for detection, true_box in pairs if true_box is not None]
    false_boxes = [detection for detection, true_box in pairs if true_box is None]
    return found, false_boxes, missed
