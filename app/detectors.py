"""Model adapters: each one wraps a model behind `detect(bgr)`."""
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from open_image_models import create_detector
from ultralytics import YOLO

from app.pipeline import DAMAGE_CONF_FLOOR, Box, Detectors

FACE_CONF = 0.4
# Detecting on the full-size photo gave false positives on gravel.
FACE_MAX_SIDE = 960
PLATE_CONF = 0.2
PLATE_MODEL = "yolo-v9-t-640-license-plate-end2end"

# model -> {class name in the model: model_class in the response}
POTHOLE_CLASSES = {"Pothole": "pothole"}
CRACK_CLASSES = {
    name: name
    for name in ("longitudinal_crack", "transverse_crack", "alligator_crack")
}


def downscale(bgr: np.ndarray, max_side: int):
    """Shrink so the longest side is <= max_side. Returns (image, factor)."""
    factor = min(1.0, max_side / max(bgr.shape[:2]))
    if factor == 1.0:
        return bgr, 1.0
    return cv2.resize(bgr, None, fx=factor, fy=factor), factor


class FaceDetector:
    def __init__(self, model_path: str):
        self._net = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), FACE_CONF
        )

    def detect(self, bgr: np.ndarray) -> list[Box]:
        small, factor = downscale(bgr, FACE_MAX_SIDE)
        height, width = small.shape[:2]
        self._net.setInputSize((width, height))
        _, rows = self._net.detect(small)
        if rows is None:
            return []
        return [
            Box(
                "face",
                float(row[-1]),
                row[0] / factor,
                row[1] / factor,
                (row[0] + row[2]) / factor,
                (row[1] + row[3]) / factor,
            )
            for row in rows
        ]


class PlateDetector:
    def __init__(self):
        self._detector = create_detector(PLATE_MODEL, conf_thresh=PLATE_CONF)

    def detect(self, bgr: np.ndarray) -> list[Box]:
        return [
            Box(
                "plate",
                float(result.confidence),
                result.bounding_box.x1,
                result.bounding_box.y1,
                result.bounding_box.x2,
                result.bounding_box.y2,
            )
            for result in self._detector.predict(bgr)
        ]


class DamageDetector:
    def __init__(self, model_path: str, classes: dict[str, str]):
        self._model = YOLO(model_path)
        self._classes = classes

    def detect(self, bgr: np.ndarray) -> list[Box]:
        result = self._model.predict(
            bgr, conf=DAMAGE_CONF_FLOOR, verbose=False
        )[0]
        boxes = []
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            label = self._classes.get(result.names[int(cls)])
            if label is not None:
                boxes.append(Box(label, conf, *xyxy))
        return boxes


def load_detectors() -> Detectors:
    """Download (or read from cache) and load the four models."""
    pothole_repo = "Samdutse/pothole-yolov8"
    crack_repo = "dronefreak/rdd2022-yolov8m"
    yunet = hf_hub_download(
        "opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx"
    )
    return Detectors(
        faces=FaceDetector(yunet),
        plates=PlateDetector(),
        potholes=DamageDetector(
            hf_hub_download(pothole_repo, "best.pt"), POTHOLE_CLASSES
        ),
        cracks=DamageDetector(
            hf_hub_download(crack_repo, "best.pt"), CRACK_CLASSES
        ),
        models={
            "damage_potholes": pothole_repo,
            "damage_cracks": crack_repo,
            "faces": "YuNet 2023mar",
            "plates": f"open-image-models {PLATE_MODEL}",
        },
    )
