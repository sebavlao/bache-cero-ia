FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # ultralytics pulls the GUI opencv-python, which needs X11 libs;
    # keep only the headless build.
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        opencv-python-headless

COPY app app
USER app
# Download the models at build time so the container starts offline.
RUN python -c "from app.detectors import load_detectors; load_detectors()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
