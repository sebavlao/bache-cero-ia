import base64

import cv2
import numpy as np

from app.detectors import downscale
from app.pipeline import Box, analyze, area_pct, blur_regions, clip_box


def test_area_pct_is_share_of_image():
    box = Box("pothole", 0.9, 0, 0, 50, 20)
    assert area_pct(box, 100, 100) == 10.0


def test_area_pct_clips_box_outside_image():
    box = Box("pothole", 0.9, -50, -50, 50, 50)
    assert area_pct(box, 100, 100) == 25.0


def test_clip_box_expands_and_clips():
    box = Box("face", 0.9, 10, 10, 30, 30)
    assert clip_box(box, 0.5, 100, 100) == (0, 0, 40, 40)


def test_blur_changes_only_the_region(noise_image):
    original = noise_image.copy()
    blur_regions(noise_image, [Box("face", 0.9, 100, 50, 160, 110)], 0)
    assert not np.array_equal(noise_image[50:110, 100:160],
                              original[50:110, 100:160])
    assert np.array_equal(noise_image[:50], original[:50])
    assert np.array_equal(noise_image[110:], original[110:])


def test_downscale_keeps_small_images(noise_image):
    image, factor = downscale(noise_image, 960)
    assert image is noise_image and factor == 1.0


def test_downscale_shrinks_longest_side():
    image, factor = downscale(np.zeros((500, 2000, 3), np.uint8), 1000)
    assert image.shape[:2] == (250, 1000) and factor == 0.5


def test_analyze_response_structure(noise_image, make_detectors):
    detectors = make_detectors(
        plates=[Box("plate", 0.8, 10, 10, 60, 30)],
        potholes=[Box("pothole", 0.81, 100, 100, 200, 180)],
        cracks=[Box("transverse_crack", 0.5, 0, 0, 30, 20)],
    )
    result = analyze(noise_image, detectors)
    assert (result["width"], result["height"]) == (300, 200)
    assert [d["model_class"] for d in result["detections"]] == [
        "pothole", "transverse_crack"]
    assert result["detections"][0] == {
        "type": "pothole", "label_es": "Bache", "model_class": "pothole",
        "confidence": 0.81, "bbox": [100, 100, 200, 180], "area_pct": 13.3,
    }
    assert result["detections"][1]["type"] == "crack"
    assert result["summary"] == {"damage_found": True, "max_confidence": 0.81}
    assert result["privacy"] == {"faces_blurred": 0, "plates_blurred": 1}
    assert set(result["timing_ms"]) == {"anonymize", "damage", "total"}
    jpeg = base64.b64decode(result["image"])
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1)
    assert decoded.shape == (200, 300, 3)


def test_analyze_without_damage(noise_image, make_detectors):
    result = analyze(noise_image, make_detectors())
    assert result["detections"] == []
    assert result["summary"] == {"damage_found": False, "max_confidence": None}


def test_damage_is_detected_on_the_anonymized_image(
        noise_image, make_detectors):
    detectors = make_detectors(faces=[Box("face", 0.9, 100, 50, 160, 110)])
    analyze(noise_image, detectors)
    seen = detectors.potholes.seen
    assert not np.array_equal(seen[50:110, 100:160],
                              noise_image[50:110, 100:160])
