"""Anonymize-then-detect pipeline.

Pure logic: detectors are injected, so nothing here loads a model.
"""
import base64
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

# Margin added around each box before blurring (fraction of its size).
PAD = {"face": 0.4, "plate": 0.15}
JPEG_QUALITY = 90
DRAW_COLOR = {"pothole": (0, 0, 255), "crack": (0, 165, 255)}  # BGR
MIN_KERNEL = 15
# Damage confidence: the models run at the floor, the response is
# filtered at the requested value (default 0.25).
DAMAGE_CONF_FLOOR = 0.05
DAMAGE_CONF_DEFAULT = 0.25

LABELS_ES = {
    "pothole": "Bache",
    "longitudinal_crack": "Grieta longitudinal",
    "transverse_crack": "Grieta transversal",
    "alligator_crack": "Grieta piel de cocodrilo",
}


@dataclass(frozen=True)
class Box:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Box]:
        """Return boxes in pixel coordinates of the given image."""


@dataclass(frozen=True)
class Detectors:
    faces: Detector
    plates: Detector
    potholes: Detector
    cracks: Detector
    models: dict[str, str]


def clip_box(box: Box, pad: float, width: int, height: int):
    """Expand a box by `pad` on every side and clip it to the image."""
    pad_x = (box.x2 - box.x1) * pad
    pad_y = (box.y2 - box.y1) * pad
    x1 = max(0, int(box.x1 - pad_x))
    y1 = max(0, int(box.y1 - pad_y))
    x2 = min(width, int(box.x2 + pad_x))
    y2 = min(height, int(box.y2 + pad_y))
    return x1, y1, x2, y2


def area_pct(box: Box, width: int, height: int) -> float:
    """Box area as a percentage of the image area."""
    x1, y1, x2, y2 = clip_box(box, 0, width, height)
    return round((x2 - x1) * (y2 - y1) / (width * height) * 100, 1)


def blur_regions(bgr: np.ndarray, boxes: list[Box], pad: float) -> None:
    """Gaussian-blur each box in place; the kernel scales with its size."""
    height, width = bgr.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = clip_box(box, pad, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        kernel = max(MIN_KERNEL, ((max(x2 - x1, y2 - y1)) // 2) | 1)
        region = bgr[y1:y2, x1:x2]
        bgr[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)


def _ms(start: float, end: float) -> int:
    return round((end - start) * 1000)


def draw_detections(bgr: np.ndarray, detections: list[dict]) -> None:
    """Draw each damage box and its label in place (demo only)."""
    thickness = max(2, max(bgr.shape[:2]) // 500)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = DRAW_COLOR[det["type"]]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
        text = f"{det['model_class']} {det['confidence']:.2f}"
        cv2.putText(
            bgr, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
            thickness / 2, color, thickness,
        )


def analyze(
    bgr: np.ndarray,
    detectors: Detectors,
    draw: bool = False,
    min_conf: float = DAMAGE_CONF_DEFAULT,
) -> dict:
    """Blur faces and plates first, then detect damage on the result.

    Damage boxes below `min_conf` are dropped; privacy thresholds are fixed.

    With `draw`, the returned image also has the damage boxes drawn.
    """
    start = time.perf_counter()
    height, width = bgr.shape[:2]

    faces = detectors.faces.detect(bgr)
    plates = detectors.plates.detect(bgr)
    clean = bgr.copy()
    blur_regions(clean, faces, PAD["face"])
    blur_regions(clean, plates, PAD["plate"])
    anonymized = time.perf_counter()

    found = detectors.potholes.detect(clean) + detectors.cracks.detect(clean)
    found = [box for box in found if box.confidence >= min_conf]
    found.sort(key=lambda box: box.confidence, reverse=True)
    detected = time.perf_counter()

    detections = [
        {
            "type": "pothole" if box.label == "pothole" else "crack",
            "label_es": LABELS_ES[box.label],
            "model_class": box.label,
            "confidence": round(box.confidence, 2),
            "bbox": list(clip_box(box, 0, width, height)),
            "area_pct": area_pct(box, width, height),
        }
        for box in found
    ]
    if draw:
        draw_detections(clean, detections)
    # Re-encoding as JPEG drops the EXIF of the original.
    ok, encoded = cv2.imencode(
        ".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    end = time.perf_counter()

    return {
        "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "width": width,
        "height": height,
        "detections": detections,
        "summary": {
            "damage_found": bool(detections),
            "max_confidence": detections[0]["confidence"]
            if detections
            else None,
        },
        "privacy": {
            "faces_blurred": len(faces),
            "plates_blurred": len(plates),
        },
        "models": detectors.models,
        "timing_ms": {
            "anonymize": _ms(start, anonymized),
            "damage": _ms(anonymized, detected),
            "total": _ms(start, end),
        },
    }
