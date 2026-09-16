"""HTTP endpoints of the API."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from PIL import Image, ImageOps

from object_detection.api.schemas import DetectionResult, DetectResponse
from object_detection.config.loader import IdentificationConfig
from object_detection.detection.inference import Detector
from object_detection.identification.identify import identify
from object_detection.identification.index import ReferenceIndex

# An APIRouter is a group of endpoints that app.py attaches to the app. It
# keeps the endpoints separate from app setup (startup, model loading), so
# each file does one job.
router = APIRouter()


def get_detector(request: Request) -> Detector:
    """Return the Detector loaded at startup (see `lifespan` in app.py).

    Used as a FastAPI dependency: an endpoint declares that it needs a
    Detector, and FastAPI calls this function to supply one. The endpoint then
    never has to know where the detector is stored.
    """
    return request.app.state.detector


def get_index(request: Request) -> ReferenceIndex:
    """Return the reference index opened at startup (see `lifespan` in app.py)."""
    return request.app.state.index


def get_identification_config(request: Request) -> IdentificationConfig:
    """Return the identification settings read from the config at startup.

    Supplied as a dependency rather than read inside the endpoint, so the
    endpoint has no idea where settings come from and tests can replace them.
    """
    return request.app.state.identification


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect objects in an image",
    description=(
        "Upload one image; returns a box for every object found in it.\n\n"
        "**Box format:** `bbox` is `[x1, y1, x2, y2]` in **absolute pixels** "
        "(top-left and bottom-right corners, origin at the top-left of the image), "
        "not normalised 0-1 values. Coordinates refer to the image after its EXIF "
        "orientation is applied, which is the size given by `image_width` and `image_height`.\n\n"
        "**Counting:** the number of entries in `detections` is the number of products found "
        "in the image. Every detection is returned, including ones that could not be identified.\n\n"
        "**Identification:** each detection's crop is matched against the reference index. "
        "`object_id`, `object_name` and `match_score` describe that match, and are `null` "
        "when no reference was similar enough, i.e. the object is not in the index."
    ),
)
def detect(
    file: Annotated[UploadFile, File(description="The image to analyse, e.g. a JPEG or PNG.")],
    detector: Annotated[Detector, Depends(get_detector)],
    index: Annotated[ReferenceIndex, Depends(get_index)],
    identification: Annotated[IdentificationConfig, Depends(get_identification_config)],
) -> DetectResponse:
    # A plain `def`, not `async def`, on purpose. Detection is slow, blocking
    # work. FastAPI runs a plain `def` endpoint in a separate worker thread, so
    # while one image is being processed the server can still accept other
    # requests. Inside `async def`, the same call would freeze the whole server
    # until it finished. Sharing one model across threads is safe: Ultralytics
    # serialises predictions with an internal lock.
    try:
        with Image.open(file.file) as uploaded:
            # Phone photos are often stored sideways, with an EXIF tag telling
            # viewers to rotate them. Given a PIL image, Ultralytics uses the
            # pixels as stored and ignores that tag (it only honours it when
            # reading a file path itself). Without this, the boxes and image
            # size would describe the sideways version, not the photo the user
            # sees. exif_transpose applies the rotation, and returns a plain
            # copy if there is nothing to rotate.
            image = ImageOps.exif_transpose(uploaded).convert("RGB")
    except OSError:
        # Pillow raises OSError (or its subclass UnidentifiedImageError) for
        # files that aren't images or are corrupt/truncated. That is the
        # client's mistake, so answer 400 Bad Request instead of crashing
        # with a 500 Internal Server Error.
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

    detections = sorted(detector.predict(image), key=lambda detection: detection.confidence, reverse=True)

    # Stage 2: crop each detection, match it against the reference index, and
    # keep only matches the threshold accepts. Every detection comes back,
    # identified or not: the number of detections is the number of products
    # found, and identification never adds or removes one.
    identifications = identify(
        image,
        detections,
        index,
        padding=identification.crop_padding,
        min_size=identification.min_crop_size,
        unknown_threshold=identification.unknown_threshold,
    )

    return DetectResponse(
        detections=[
            DetectionResult(
                bbox=list(result.detection.bbox),
                confidence=result.detection.confidence,
                object_id=result.object_id,
                object_name=result.object_name,
                match_score=result.match_score,
            )
            for result in identifications
        ],
        image_width=image.width,
        image_height=image.height,
    )
