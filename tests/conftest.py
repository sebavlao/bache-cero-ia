import cv2
import numpy as np
import pytest

from app.pipeline import Detectors


class FakeDetector:
    """Returns fixed boxes and remembers the last image it received."""

    def __init__(self, boxes=()):
        self.boxes = list(boxes)
        self.seen = None

    def detect(self, bgr):
        self.seen = bgr.copy()
        return self.boxes


@pytest.fixture
def noise_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)


@pytest.fixture
def make_detectors():
    def _make(faces=(), plates=(), potholes=(), cracks=()):
        return Detectors(
            faces=FakeDetector(faces),
            plates=FakeDetector(plates),
            potholes=FakeDetector(potholes),
            cracks=FakeDetector(cracks),
            models={"faces": "fake"},
        )

    return _make


@pytest.fixture
def jpeg_bytes(noise_image):
    return cv2.imencode(".jpg", noise_image)[1].tobytes()
