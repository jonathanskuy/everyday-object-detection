"""Run the trained detector on an image and return plain detections.

Usage:
    from object_detection.config.loader import load_config
    from object_detection.detection.inference import Detector

    cfg = load_config()
    detector = Detector(cfg.inference.weights, cfg.inference.conf)
    for detection in detector.predict("photo.jpg"):
        print(detection.bbox, detection.confidence)

This module is the boundary between Ultralytics and the rest of the project:
Ultralytics' Results objects stay inside it, and everything downstream
(notebooks, the API later) only sees Detection. Swapping or upgrading the
detector then touches this file alone.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


# frozen=True: a detection is a result, not something to edit afterwards.
@dataclass(frozen=True)
class Detection:
    """One detected item.

    `bbox` is (x1, y1, x2, y2): the top-left and bottom-right corners in
    ABSOLUTE PIXELS of the original image, not normalised 0-1 values. This is
    the same format as `bbox` in the API response. It is also not the
    (x_center, y_center, width, height) format of the YOLO label files.
    """

    bbox: tuple[float, float, float, float]
    confidence: float


class Detector:
    """Loads the trained weights once, then detects items in any number of images.

    A class rather than a function because loading the weights costs far more
    than a single prediction. A notebook creates one Detector and reuses it;
    the API will create one at startup and share it across requests.
    """

    def __init__(self, weights: Path, conf: float):
        self._model = YOLO(str(weights))
        self.conf = conf

    def predict(self, image: str | Path | Image.Image) -> list[Detection]:
        """Return every detection in `image` with confidence >= `self.conf`.

        `image` is a file path or a PIL image. NumPy arrays are deliberately
        not accepted: Ultralytics treats them as BGR (OpenCV's channel order),
        so passing an RGB array silently swaps red and blue.
        """
        # Ultralytics returns one Results per input image; we pass exactly one.
        # imgsz is not passed, as in the notebook: Ultralytics then uses the
        # image size stored in the checkpoint, i.e. the one it was trained at.
        result = self._model(image, conf=self.conf, verbose=False)[0]

        # The tensors may live on the GPU; .cpu().tolist() turns them into
        # plain Python floats that the rest of the project can use directly.
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        return [
            Detection(bbox=tuple(bbox), confidence=confidence)
            for bbox, confidence in zip(boxes, confidences)
        ]
