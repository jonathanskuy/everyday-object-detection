"""The shape of the API's responses, as Pydantic models.

FastAPI uses these models in two ways:
  - every response is checked and converted to JSON through them, so the API
    can never return a field that isn't declared here, or miss one that is;
  - they generate the OpenAPI schema shown at /docs, so the descriptions below
    ARE the API's public documentation.

The response shape is fixed (see CLAUDE.md). Stage 2 fills in the three
reserved fields; it must not add, remove or rename anything.
"""

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """One detected item in the uploaded image."""

    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description=(
            "Bounding box as [x1, y1, x2, y2]: the top-left and bottom-right corners, "
            "in ABSOLUTE PIXELS of the image, with the origin at the image's top-left corner. "
            "Not normalised to 0-1, and not [x_center, y_center, width, height]. "
            "Coordinates refer to the image after its EXIF orientation is applied, "
            "i.e. the image as it is normally displayed, sized image_width x image_height."
        ),
        examples=[[902.3, 728.1, 1012.6, 942.0]],
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Detector confidence that this box contains an object, from 0 to 1.",
        examples=[0.91],
    )
    # Reserved for Stage 2 (identification). Always null for now: returning
    # them already means clients are written against the final shape, and
    # Stage 2 only changes their values, never the structure.
    object_id: str | None = Field(
        default=None,
        description="Identifier of the matched reference object. Reserved for Stage 2; currently always null.",
    )
    object_name: str | None = Field(
        default=None,
        description="Name of the matched reference object. Reserved for Stage 2; currently always null.",
    )
    match_score: float | None = Field(
        default=None,
        description="Similarity score of the identification match. Reserved for Stage 2; currently always null.",
    )


class DetectResponse(BaseModel):
    """Everything detected in one uploaded image."""

    detections: list[DetectionResult] = Field(
        description="All detected items, highest confidence first. An empty list if nothing was detected.",
    )
    image_width: int = Field(
        description="Width in pixels of the image the bbox coordinates refer to (after EXIF orientation).",
        examples=[1920],
    )
    image_height: int = Field(
        description="Height in pixels of the image the bbox coordinates refer to (after EXIF orientation).",
        examples=[1080],
    )
