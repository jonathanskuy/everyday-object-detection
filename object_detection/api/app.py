"""The FastAPI application: creates the app and loads the detector once at startup.

Run from the repository root:
    uvicorn object_detection.api.app:app

Then open http://127.0.0.1:8000/docs for interactive documentation, where
images can be uploaded to /detect straight from the browser.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from PIL import Image

from object_detection.api.routes import router
from object_detection.config.loader import load_config
from object_detection.detection.inference import Detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up shared resources before the server accepts requests.

    FastAPI runs the code before `yield` once, at startup, and the code after
    `yield` once, at shutdown. Loading the model here rather than inside the
    endpoint means it is loaded one time, not on every request.
    """
    cfg = load_config()
    detector = Detector(cfg.inference.weights, cfg.inference.conf)

    # Warm-up prediction on a blank image. The first prediction is slow
    # (Ultralytics builds its predictor, and the GPU initialises), so doing it
    # here moves that delay from the first user's request to startup. It also
    # guarantees the predictor exists before requests can arrive at the same
    # time. The image size doesn't matter: every input is resized for the model.
    detector.predict(Image.new("RGB", (640, 640)))

    # app.state is FastAPI's place for objects shared across all requests.
    app.state.detector = detector
    yield
    # Nothing to clean up at shutdown: the model is freed with the process.


app = FastAPI(
    title="Everyday Object Detection",
    description=(
        "Stage 1 of a two-stage recognition system: a class-agnostic YOLO detector finds "
        "where objects are in an image. Stage 2 (identification by vector retrieval) is "
        "not implemented yet, so identification fields in responses are null."
    ),
    lifespan=lifespan,
)
app.include_router(router)
