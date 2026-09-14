"""Tests for the /detect endpoint.

The real model is replaced by a fake detector, so these tests check the API
layer itself (upload handling, EXIF rotation, status codes, response shape)
quickly and without weights or a GPU.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from object_detection.api.app import app
from object_detection.api.routes import get_detector
from object_detection.detection.inference import Detection


class FakeDetector:
    """Returns fixed detections and records the size of every image it receives."""

    def __init__(self, detections):
        self.detections = detections
        self.received_sizes = []

    def predict(self, image):
        self.received_sizes.append(image.size)
        return self.detections


@pytest.fixture
def fake_detector():
    # Deliberately NOT sorted by confidence, to test that the API sorts.
    detector = FakeDetector([
        Detection(bbox=(1.0, 2.0, 3.0, 4.0), confidence=0.4),
        Detection(bbox=(10.0, 20.0, 30.0, 40.0), confidence=0.9),
    ])
    # dependency_overrides is FastAPI's way to swap a dependency in tests:
    # wherever an endpoint asks for get_detector, it gets the fake instead.
    app.dependency_overrides[get_detector] = lambda: detector
    yield detector
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    # Used without `with`, TestClient does not run the app's lifespan, so the
    # real model is never loaded.
    return TestClient(app)


def jpeg_bytes(size, exif_orientation=None):
    """Encode a blank image as JPEG in memory, optionally with an EXIF orientation tag."""
    buffer = io.BytesIO()
    image = Image.new("RGB", size, "white")
    if exif_orientation is None:
        image.save(buffer, format="JPEG")
    else:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def post_image(client, content, filename="photo.jpg"):
    return client.post("/detect", files={"file": (filename, content, "image/jpeg")})


def test_response_has_the_agreed_shape(client, fake_detector):
    response = post_image(client, jpeg_bytes((200, 100)))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"detections", "image_width", "image_height"}
    for detection in body["detections"]:
        assert set(detection) == {"bbox", "confidence", "object_id", "object_name", "match_score"}
        # Reserved for Stage 2: present, and null.
        assert detection["object_id"] is None
        assert detection["object_name"] is None
        assert detection["match_score"] is None


def test_detections_are_returned_highest_confidence_first(client, fake_detector):
    body = post_image(client, jpeg_bytes((200, 100))).json()
    assert [d["confidence"] for d in body["detections"]] == [0.9, 0.4]
    assert body["detections"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_image_size_is_reported(client, fake_detector):
    body = post_image(client, jpeg_bytes((200, 100))).json()
    assert (body["image_width"], body["image_height"]) == (200, 100)


def test_exif_rotated_photo_is_turned_upright_before_detection(client, fake_detector):
    # Stored 200 wide x 100 high, tagged "rotate 90 degrees to display" (6):
    # displayed upright it is 100 wide x 200 high.
    body = post_image(client, jpeg_bytes((200, 100), exif_orientation=6)).json()
    assert (body["image_width"], body["image_height"]) == (100, 200)
    assert fake_detector.received_sizes == [(100, 200)]


def test_no_detections_gives_an_empty_list(client):
    app.dependency_overrides[get_detector] = lambda: FakeDetector([])
    try:
        body = post_image(client, jpeg_bytes((200, 100))).json()
    finally:
        app.dependency_overrides.clear()
    assert body["detections"] == []


def test_a_file_that_is_not_an_image_is_rejected_with_400(client, fake_detector):
    response = post_image(client, b"this is not an image")
    assert response.status_code == 400
    assert fake_detector.received_sizes == []


def test_a_request_without_a_file_is_rejected_with_422(client, fake_detector):
    assert client.post("/detect").status_code == 422


def test_bbox_format_is_documented_in_openapi(client):
    schema = client.get("/openapi.json").json()
    description = schema["components"]["schemas"]["DetectionResult"]["properties"]["bbox"]["description"]
    assert "ABSOLUTE PIXELS" in description
    assert "[x1, y1, x2, y2]" in description
