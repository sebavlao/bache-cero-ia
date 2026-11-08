# bache-cero-ia

Microservicio de visión por computadora de **Bache Cero**. Recibe una imagen de una calle, difumina los rostros y las patentes que aparecen en ella y, sobre esa misma imagen, detecta baches y grietas.

Su función es analizar y devolver los resultados de los modelos: qué se detectó, con qué nivel de confianza y en qué ubicación de la imagen. No intenta determinar la gravedad de los daños ni establecer prioridades de reparación. Esas decisiones corresponden al sistema que utiliza el servicio y a las personas encargadas de revisarlas.

## Cómo ejecutar el proyecto

### Local

Se necesita Python 3.12.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app
```

En el primer inicio se descargan los modelos desde Hugging Face (unos 80 MB en total). Después quedan almacenados en la caché local y no es necesario volver a descargarlos.

La API cuenta con documentación interactiva, disponible una vez iniciado el servidor:

[http://localhost:8000/docs](http://localhost:8000/docs)

Para ejecutar los tests y revisar el estilo del código:

```bash
pip install -r requirements-dev.txt
pytest -q
flake8 app tests
```

### Docker

Si preferís levantar el servicio con Docker:

```bash
docker build -t bache-cero-ia .
docker run -p 8000:8000 bache-cero-ia
```

Durante la construcción de la imagen se descargan los modelos y sus dependencias. Esto permite iniciar el contenedor sin necesidad de conexión a internet.

La imagen ocupa aproximadamente 2,5 GB, principalmente por PyTorch.

### Ejemplo

Para probar el análisis desde la terminal:

```bash
curl -F file=@foto.jpg localhost:8000/v1/analyze
```

Con los parámetros opcionales (`conf` y `draw`, ver [API](#post-v1analyze)):

```bash
curl -F file=@foto.jpg "localhost:8000/v1/analyze?conf=0.1&draw=true"
```

## API

El servicio dispone de dos endpoints.

### `GET /health`

Permite comprobar que el servicio está funcionando y que los modelos se encuentran cargados.

```json
{
  "status": "ok",
  "models_loaded": true
}
```

### `POST /v1/analyze`

Es el endpoint principal del servicio. Recibe una imagen y devuelve el resultado del procesamiento.

La imagen debe enviarse mediante `multipart/form-data`, utilizando el campo `file`. Se aceptan archivos JPEG, PNG y WebP de hasta 10 MB.

Además del archivo, se pueden utilizar dos parámetros opcionales:

- **`draw=true`**: agrega las cajas de detección de daños sobre la imagen devuelta. Está pensado principalmente para visualizar los resultados durante las pruebas y demostraciones. No modifica el JSON.
- **`conf`**: permite ajustar la confianza mínima requerida para aceptar una detección de daño. Admite valores entre `0.05` y `1`, con `0.25` como valor predeterminado. No modifica los umbrales de detección de rostros ni de patentes, que se mantienen fijos.

Al bajar el valor de `conf` pueden aparecer más detecciones, pero también aumenta la posibilidad de obtener falsos positivos. Por eso, salvo que se esté probando el comportamiento del modelo, conviene mantener el valor predeterminado.

**Ejemplo de respuesta (`200 OK`):**

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
  "summary": {
    "damage_found": true,
    "max_confidence": 0.81
  },
  "privacy": {
    "faces_blurred": 1,
    "plates_blurred": 0
  },
  "models": {
    "damage_potholes": "Samdutse/pothole-yolov8",
    "damage_cracks": "dronefreak/rdd2022-yolov8m",
    "faces": "YuNet 2023mar",
    "plates": "open-image-models yolo-v9-t-640-license-plate-end2end"
  },
  "timing_ms": {
    "anonymize": 45,
    "damage": 310,
    "total": 380
  }
}
```

**Campos de la respuesta:**

- `image`: imagen procesada en formato JPEG y codificada en Base64.
- `width` y `height`: dimensiones originales de la imagen.
- `detections`: lista de daños encontrados y sus respectivas características.
- `type`: categoría general del daño. Puede ser `pothole` o `crack`.
- `label_es`: nombre de la detección en español.
- `model_class`: clase original devuelta por el modelo. Para las grietas puede ser `longitudinal_crack`, `transverse_crack` o `alligator_crack`.
- `confidence`: nivel de confianza de la detección.
- `bbox`: coordenadas de la caja delimitadora en píxeles, siguiendo el formato `[x1, y1, x2, y2]`.
- `area_pct`: porcentaje de la superficie de la imagen que ocupa la caja delimitadora. No representa necesariamente el área real del daño.
- `summary`: indica si se encontraron daños y cuál fue la confianza máxima.
- `privacy`: cantidad de rostros y patentes difuminados durante el procesamiento.
- `models`: identificación de los modelos utilizados.
- `timing_ms`: tiempo de procesamiento de cada etapa y tiempo total, expresado en milisegundos.

Las detecciones de daños se filtran según el valor de `conf` enviado en la solicitud. Si no se especifica, se utiliza `0.25`.

**Códigos de error:**

| Código | Motivo                                      |
| ------ | ------------------------------------------- |
| `413`  | El archivo supera el límite de 10 MB.       |
| `415`  | El formato de imagen no está soportado.     |
| `422`  | La imagen está corrupta o no se puede leer. |

## Modelos utilizados

Para cubrir las distintas tareas del servicio se utilizan modelos independientes. Cada uno se encarga de una parte del procesamiento:

| Tarea                 | Modelo                                                    | Licencia                           |
| --------------------- | --------------------------------------------------------- | ---------------------------------- |
| Detección de baches   | `Samdutse/pothole-yolov8`                                 | Sin licencia declarada (solo demo) |
| Detección de grietas  | `dronefreak/rdd2022-yolov8m`                              | AGPL-3.0                           |
| Detección de rostros  | YuNet (OpenCV Zoo)                                        | MIT                                |
| Detección de patentes | `yolo-v9-t-640-license-plate-end2end` (open-image-models) | MIT                                |

La elección de los modelos y las pruebas realizadas durante el desarrollo están documentadas en el siguiente notebook:

[Bache Cero · Pruebas preliminares de IA](https://colab.research.google.com/drive/1FR5wLn4YrKEsEEIPMIywQe4rPIM4hx8d?usp=sharing)

Allí se encuentra información sobre el origen de cada modelo, sus datos de entrenamiento, las evaluaciones realizadas, las limitaciones detectadas y las diferencias respecto de sus fichas originales. También se detallan las licencias y consideraciones de uso.

## Privacidad

Las imágenes se procesan en memoria. El servicio no guarda los archivos originales en disco ni registra su contenido en los logs.

Una vez terminado el análisis, la imagen de salida se vuelve a codificar en JPEG y se eliminan sus metadatos EXIF. De esta manera, la respuesta contiene la versión procesada y no el archivo original.

El difuminado depende de la detección correcta de rostros y patentes, por lo que no debe interpretarse como una garantía de anonimización absoluta.

## Licencias y créditos

El repositorio se distribuye bajo la licencia **AGPL-3.0**, debido al uso de Ultralytics.

Además, se utilizan los siguientes proyectos y modelos:

- [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- `dronefreak/rdd2022-yolov8m` (AGPL-3.0), entrenado con el conjunto de datos RDD2022 (CC BY 4.0, Arya et al.).
- YuNet, incluido en [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (MIT).
- [open-image-models](https://github.com/ankandrew/open-image-models), desarrollado por ankandrew (MIT). Se conserva el aviso de copyright correspondiente.
- `Samdutse/pothole-yolov8`: no declara una licencia de uso. Por este motivo, se utiliza únicamente en la demo y las pruebas preliminares, y no se incorpora como parte del producto.

El detalle de las licencias, los créditos y las condiciones de uso de cada modelo se encuentra en la sección **Créditos y licencias** del notebook de pruebas.
