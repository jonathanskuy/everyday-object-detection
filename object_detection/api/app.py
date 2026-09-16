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
from object_detection.identification.index import create_index


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

    # The reference index, loaded once for the same reason as the detector.
    # With Qdrant in local mode this LOCKS its folder for as long as the
    # server runs, so build_reference_index.py cannot run at the same time.
    # That is the point at which Qdrant moves to a server (see the config).
    index = create_index(cfg.identification)

    # app.state is FastAPI's place for objects shared across all requests.
    app.state.detector = detector
    app.state.index = index
    app.state.identification = cfg.identification
    try:
        yield
    finally:
        # Releases the local-mode lock, so the index can be rebuilt once the
        # server stops. The model is freed with the process.
        index.close()


app = FastAPI(
    title="Everyday Object Detection",
    description=(
        "Two-stage recognition: a class-agnostic YOLO detector finds where objects are, "
        "then each detection's crop is embedded and matched against a reference index to "
        "decide which object it is. Items whose best match is too weak are returned as "
        "detections with null identification fields."
    ),
    lifespan=lifespan,
)
app.include_router(router)
