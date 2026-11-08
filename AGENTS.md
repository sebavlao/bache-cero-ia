# AGENTS.md — bache-cero-ia

Vision microservice of **Bache Cero**. It receives a street photo, blurs faces and license plates, detects potholes and cracks on the blurred image and returns what the models see. Read `docs/PRD.md` first (API contract, rules, models, acceptance criteria) and `README.md` for usage.

## Stack

Python 3.12 · FastAPI + Uvicorn · Ultralytics (YOLOv8, damage) · OpenCV `FaceDetectorYN` (YuNet, faces) · `open-image-models[onnx]` (plates) · `huggingface_hub` · pytest · flake8 · Docker. CPU only.

## Layout

```
app/main.py        # create_app(): /, /health, /v1/analyze, 413/415/422 validation
app/pipeline.py    # Box, Detectors, analyze(): pure logic, injected detectors
app/detectors.py   # adapters for the four models + load_detectors()
app/schemas.py     # Pydantic response models
tests/             # fake detectors (no model loading) + TestClient
```

## Rules

- **Pipeline order**: anonymize first, then detect damage on the anonymized image.
- **The service perceives, it does not decide**: severity, human review, duplicates and priority belong to the consumer (see `docs/PRD.md` §5.1).
- **Privacy**: the original image is never written to disk or logs; the output is re-encoded as JPEG (no EXIF).
- **Low thresholds for faces and plates** (blur more rather than less). They are not configurable; only the damage threshold is (`?conf=`).
- **Plate box only**, no OCR.
- **Validate at the edge**: size <= 10 MB, JPEG/PNG/WebP, decodable image (413/415/422).
- **One source per class**: potholes only from Samdutse, cracks only from dronefreak (its pothole class is ignored), so there are never two pothole boxes on the same area.
- Inference is serialized with a lock; models are loaded once at startup.

## Models

| Use | Source | File | Classes |
|---|---|---|---|
| Potholes | Hugging Face `Samdutse/pothole-yolov8` | `best.pt` | `{0: 'Pothole'}` |
| Cracks | Hugging Face `dronefreak/rdd2022-yolov8m` | `best.pt` | `longitudinal_crack`, `transverse_crack`, `alligator_crack` (and `pothole`, ignored) |
| Faces | Hugging Face `opencv/face_detection_yunet` | `face_detection_yunet_2023mar.onnx` | — |
| Plates | `open-image-models` | `yolo-v9-t-640-license-plate-end2end` | `license_plate` |

- Samdutse declares no license: demo use only.
- Docker downloads the models at build time so the container starts offline.

## Tuned parameters

- YuNet runs on a copy downscaled to 960 px (longest side) and boxes are scaled back; at full resolution large phone photos produced false faces on gravel texture. `FACE_CONF = 0.4`.
- Padding per type: `PAD = {"face": 0.4, "plate": 0.15}`. `PLATE_CONF = 0.2`.
- Gaussian blur with an odd kernel proportional to the region: `max(15, longest_side // 2 | 1)`.
- Damage models run at a 0.05 floor; the response is filtered at `conf` (default 0.25).

## Conventions

- Code, identifiers and comments in English; documentation in neutral Spanish.
- Strict PEP 8: `flake8 app tests` with no warnings (79 columns).
- Minimal changes: no abstractions or options nobody asked for.
- Non-trivial logic (validation, box clipping, `area_pct`) comes with a test; tests use fake detectors.
- Conventional Commits.
- Never commit test photos: use an ignored `samples/` folder, and synthetic images in automated tests.

## Commands

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest -q && flake8 app tests
curl -F file=@photo.jpg localhost:8000/v1/analyze
docker build -t bache-cero-ia . && docker run -p 8000:8000 bache-cero-ia
```
