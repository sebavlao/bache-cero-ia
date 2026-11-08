# PRD — Bache Cero · Microservicio de visión

> Bache Cero es una plataforma de apoyo a la decisión para el área municipal de mantenimiento vial de Posadas. Este repo contiene solo el servicio de visión que consume el backend.

## 1. Problema que resuelve

El backend necesita, por cada foto que suben vecinos y cuadrillas:
1. Una versión **anonimizada** (rostros y patentes difuminados, sin metadatos EXIF), que es la única que se guarda (Cloudinary) y se muestra.
2. Las **detecciones de daño** (tipo, confianza, ubicación en la imagen, tamaño relativo).

El servicio hace exactamente eso: recibe la imagen y devuelve lo que ven los modelos. **No decide**: la gravedad, la revisión humana, los duplicados y la prioridad son reglas de negocio del backend.

## 2. Alcance

### Dentro del alcance
- API HTTP en Python (FastAPI) con cuatro modelos de visión: dos de daños, rostros y patentes.
- Anonimización: detección de rostros y patentes + GaussianBlur.
- Detección y clasificación de baches y grietas.

### Fuera del alcance
- Gravedad, revisión humana, duplicados (PostGIS) y prioridad: backend NestJS. Sugerencia de reglas en §5.1.
- Almacenamiento, autenticación, roles, tickets, mapa.
- Lectura de patentes (OCR): solo se necesita el recuadro para difuminar.
- Una clase propia de hundimiento: ningún modelo la tiene. Los que rompen la calzada se espera que salgan como bache (no probado).
- Entrenamiento o ajuste de modelos.

## 3. Consumidor

El **backend NestJS** (llamada HTTP interna). Indirectamente: personal municipal (ve fotos anonimizadas) y vecinos (privacidad).

## 4. Pipeline

1. Validar el archivo (tamaño, tipo, decodificable).
2. **Anonimizar**: detectar rostros (YuNet, sobre una copia reducida a 960 px de lado mayor para evitar falsos positivos en fotos grandes) y patentes (YOLOv9) con umbrales bajos, y aplicar GaussianBlur sobre cada caja ampliada (40 % en rostros, 15 % en patentes).
3. **Detectar daños sobre la imagen ya anonimizada**: baches con `Samdutse/pothole-yolov8` y grietas con `dronefreak/rdd2022-yolov8m`. De dronefreak se ignoran sus baches: en fotos locales no los detecta (domain shift) y así nunca hay dos cajas de bache sobre la misma zona.
4. Re-codificar la imagen anonimizada en JPEG (sin EXIF) y responder.

## 5. Requisitos funcionales

### RF1 — `GET /health`
`200` con `{ "status": "ok", "models_loaded": true }` cuando los cuatro modelos están cargados.

### RF2 — `POST /v1/analyze`
- Entrada: `multipart/form-data`, campo `file` (JPEG, PNG o WebP).
- Respuesta `200`:

```json
{
  "image": "<JPEG base64, difuminado y sin EXIF>",
  "width": 1280,
  "height": 960,
  "detections": [
    {
      "type": "pothole",
      "label_es": "Bache",
      "model_class": "pothole",
      "confidence": 0.81,
      "bbox": [412, 530, 690, 720],
      "area_pct": 4.3
    }
  ],
  "summary": { "damage_found": true, "max_confidence": 0.81 },
  "privacy": { "faces_blurred": 1, "plates_blurred": 0 },
  "models": {
    "damage_potholes": "Samdutse/pothole-yolov8",
    "damage_cracks": "dronefreak/rdd2022-yolov8m",
    "faces": "YuNet 2023mar",
    "plates": "open-image-models yolo-v9-t-640-license-plate-end2end"
  },
  "timing_ms": { "anonymize": 45, "damage": 310, "total": 380 }
}
```

- `type` ∈ `pothole` | `crack`. `model_class` conserva la clase original: `pothole`, `longitudinal_crack`, `transverse_crack`, `alligator_crack`.
- `bbox` en píxeles `[x1, y1, x2, y2]`; `area_pct` = área de la caja / área de la imagen × 100.
- Umbral de confianza de daño: 0.25 por defecto; se puede cambiar con `?conf=` (0.05 a 1). Por debajo del umbral no se devuelve la detección. Los umbrales de rostros y patentes son fijos. Fuera de ese umbral técnico no se aplican reglas de negocio.
- Errores: `413` archivo > 10 MB; `415` tipo no soportado; `422` imagen corrupta o ilegible.

### 5.1 Referencia para el backend (no lo implementa este servicio)
Sugerencia a acordar con el equipo de backend y calibrar con el municipio. Vive en el backend para poder ajustarse sin redesplegar el servicio de visión.

**Revisión humana**: marcar para revisión si no hay detecciones o si `summary.max_confidence` < 0.4.

**Gravedad**: por detección, según `model_class` y `area_pct`; la del reporte es la peor. **Valores iniciales sin validar**:

| Tipo | Leve | Moderado | Grave |
|---|---|---|---|
| Bache | `area_pct` < 1 | 1 – 5 | > 5 |
| Grieta piel de cocodrilo | < 3 | 3 – 10 | > 10 |
| Grieta longitudinal / transversal | < 5 | ≥ 5 | — |

Limitación: sin referencia de escala, el tamaño en la foto depende de la distancia y el ángulo.

Próxima etapa: si el detector se reentrena con clases que incluyen la gravedad (por ejemplo `bache_grave`), el backend pasa de aplicar la regla a leer `model_class`. El contrato no cambia.

## 6. Requisitos no funcionales

- **Privacidad**: la imagen original se procesa solo en memoria; nunca se escribe en disco ni en logs. La salida se re-codifica (elimina EXIF: GPS, fecha, modelo de celular).
- **Seguridad en el borde**: validar tamaño y tipo antes de decodificar; rechazar lo que PIL/OpenCV no puedan abrir.
- **CPU** por defecto (sin GPU). Exportar YOLO a ONNX queda como mejora.
- **Modelos cargados una sola vez** al iniciar el proceso.
- **Docker** reproducible.
- **Licencia del repo: AGPL-3.0**, obligatoria por Ultralytics.

## 7. Modelos

| Tarea | Modelo | Origen | Licencia | Nota |
|---|---|---|---|---|
| Baches | `Samdutse/pothole-yolov8` (YOLOv8s), archivo `best.pt`, clase `Pothole` | Hugging Face | Sin licencia declarada | Solo para la demo; no se redistribuye |
| Grietas | `dronefreak/rdd2022-yolov8m` (YOLOv8m), archivo `best.pt` | Hugging Face, entrenado con RDD2022 (CC BY 4.0) | AGPL-3.0 | Se ignora su clase bache |
| Rostros | YuNet `face_detection_yunet_2023mar.onnx` | Hugging Face `opencv/face_detection_yunet` (OpenCV Zoo) | MIT | Detectar sobre copia de 960 px |
| Patentes | `yolo-v9-t-640-license-plate-end2end` | paquete `open-image-models[onnx]` (ankandrew), descarga automática | MIT (conservar aviso de copyright) | Datos de entrenamiento no declarados |

Los modelos se descargan en el build de Docker, no en cada arranque. Ver el código de carga en `app/detectors.py`.

Las métricas publicadas por los autores no se reprodujeron. El detalle de la elección de cada modelo está en el notebook enlazado en el `README.md`.

## 8. Criterios de aceptación

- `pytest` pasa: health; analyze con imagen de prueba (estructura de respuesta completa); 415 con archivo no imagen; 413 con archivo grande; test unitario del cálculo de `area_pct`.
- `curl -F file=@foto.jpg localhost:8000/v1/analyze` con fotos reales devuelve detecciones y una imagen con patentes/rostros visiblemente difuminados.
- La imagen devuelta no contiene EXIF.
- Una foto vertical de celular (con orientación en EXIF) se analiza y se devuelve con la orientación correcta.
- Con una foto de celular de alta resolución (5184 px) con una persona de perfil se difumina solo la cara, no el antebrazo ni el piso.
- `docker build` + `docker run` funcionan y `/health` responde.

## 9. Riesgos

- Detector de patentes con datos de entrenamiento desconocidos: puede fallar con patentes Mercosur, lejanas o en ángulo.
- YuNet falla con rostros chicos, de perfil o con poca luz.
- Domain shift: los modelos de daños no se ajustaron con fotos de Posadas.
- AGPL-3.0: que el aislamiento por HTTP proteja al backend es una cuestión legal, no técnica.

## 10. Preguntas abiertas

- Acordar con el backend la implementación de §5.1.

## 11. Fuentes

- YuNet: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- open-image-models: https://github.com/ankandrew/open-image-models
- dronefreak: https://huggingface.co/dronefreak/rdd2022-yolov8m · Samdutse: https://huggingface.co/Samdutse/pothole-yolov8
- RDD2022: https://arxiv.org/abs/2209.08538
