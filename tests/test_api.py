import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import MAX_BYTES, create_app
from app.pipeline import Box


def post(client, data, name="foto.jpg"):
    return client.post("/v1/analyze", files={"file": (name, data)})


def test_health_and_index(make_detectors):
    with TestClient(create_app(make_detectors())) as client:
        assert client.get("/health").json() == {
            "status": "ok", "models_loaded": True}
        assert client.get("/").json()["health"] == "/health"


def test_analyze_ok(make_detectors, jpeg_bytes):
    detectors = make_detectors(potholes=[Box("pothole", 0.8, 0, 0, 30, 30)])
    with TestClient(create_app(detectors)) as client:
        response = post(client, jpeg_bytes)
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["damage_found"] is True
    assert body["detections"][0]["type"] == "pothole"


def _decode(body):
    data = base64.b64decode(body["image"])
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), int)


def test_draw_adds_boxes_and_default_does_not(make_detectors, jpeg_bytes):
    detectors = make_detectors(potholes=[Box("pothole", 0.8, 5, 5, 40, 40)])
    with TestClient(create_app(detectors)) as client:
        plain = post(client, jpeg_bytes).json()
        drawn = client.post(
            "/v1/analyze?draw=true", files={"file": ("f.jpg", jpeg_bytes)}
        ).json()
    assert plain["detections"] == drawn["detections"]
    assert np.abs(_decode(plain) - _decode(drawn)).max() > 50


def test_conf_filters_damage_and_is_validated(make_detectors, jpeg_bytes):
    detectors = make_detectors(
        potholes=[Box("pothole", 0.3, 0, 0, 30, 30),
                  Box("pothole", 0.1, 40, 40, 60, 60)])

    def run(query):
        with TestClient(create_app(detectors)) as client:
            return client.post(
                f"/v1/analyze{query}", files={"file": ("f.jpg", jpeg_bytes)})

    assert len(run("").json()["detections"]) == 1
    assert len(run("?conf=0.05").json()["detections"]) == 2
    assert run("?conf=0.5").json()["detections"] == []
    assert run("?conf=0.01").status_code == 422
    assert run("?conf=2").status_code == 422


def test_non_image_is_415(make_detectors):
    with TestClient(create_app(make_detectors())) as client:
        assert post(client, b"hello", "a.txt").status_code == 415


def test_large_file_is_413(make_detectors):
    with TestClient(create_app(make_detectors())) as client:
        assert post(client, b"0" * (MAX_BYTES + 1)).status_code == 413


def test_corrupt_image_is_422(make_detectors):
    with TestClient(create_app(make_detectors())) as client:
        assert post(client, b"\xff\xd8\xff" + b"garbage").status_code == 422


def test_output_has_no_exif_and_input_orientation_is_applied(
        make_detectors):
    # 40x20 image tagged "rotate 90": decoded it must be 20x40.
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "gray").save(buffer, "JPEG", exif=exif)
    with TestClient(create_app(make_detectors())) as client:
        body = post(client, buffer.getvalue()).json()
    assert (body["width"], body["height"]) == (20, 40)
    out = Image.open(io.BytesIO(base64.b64decode(body["image"])))
    assert len(out.getexif()) == 0
    assert out.size == (20, 40)
