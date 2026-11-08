"""HTTP layer: validation at the edge, then the pipeline."""
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile

from app.pipeline import (
    DAMAGE_CONF_DEFAULT,
    DAMAGE_CONF_FLOOR,
    Detectors,
    analyze,
)
from app.schemas import AnalyzeResponse, Health

MAX_BYTES = 10 * 1024 * 1024


def _is_supported_image(data: bytes) -> bool:
    """Check magic bytes: JPEG, PNG or WebP."""
    return (
        data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"\x89PNG\r\n\x1a\n")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


def create_app(detectors: Detectors | None = None) -> FastAPI:
    """Build the app; pass `detectors` to skip loading the real models."""
    # The models are not documented as thread-safe and inference is
    # CPU-bound anyway, so requests are processed one at a time.
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if detectors is not None:
            app.state.detectors = detectors
        else:
            from app.detectors import load_detectors

            app.state.detectors = load_detectors()
        yield

    app = FastAPI(title="Bache Cero - vision service", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> dict[str, str]:
        return {"docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=Health)
    def health() -> Health:
        loaded = getattr(app.state, "detectors", None) is not None
        return Health(status="ok", models_loaded=loaded)

    @app.post("/v1/analyze", response_model=AnalyzeResponse)
    def analyze_image(
        file: UploadFile,
        draw: bool = False,
        conf: float = Query(DAMAGE_CONF_DEFAULT, ge=DAMAGE_CONF_FLOOR, le=1),
    ) -> dict:
        data = file.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File larger than 10 MB")
        if not _is_supported_image(data):
            raise HTTPException(415, "Use a JPEG, PNG or WebP image")
        # imdecode applies the EXIF orientation, like imread.
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(422, "Image is corrupt or unreadable")
        with lock:
            return analyze(bgr, app.state.detectors, draw, conf)

    return app


app = create_app()
