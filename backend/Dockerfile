# Veritas backend image: data prep CLI + inference API + Celery worker.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg/libsndfile are needed for video frame extraction and audio decoding
# (Phases 4-5). Kept in the base image so the same image serves all roles.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
COPY src ./src

# The `api` extra pulls FastAPI/ONNX Runtime/Redis/Celery used by the service.
RUN pip install --upgrade pip && pip install ".[api]"

# Non-root runtime user.
RUN useradd --create-home --uid 1000 veritas
USER veritas

EXPOSE 8000

# Default to the API; compose overrides `command` for the Celery worker.
CMD ["uvicorn", "veritas.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
