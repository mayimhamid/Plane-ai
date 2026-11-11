# Plane AI model initializer: downloads + verifies the GGUF model once, then exits.
# Standard library only, so no pip install step.
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLANE_AI_MODEL_STORAGE_PATH=/models

RUN groupadd --system --gid 10001 planeai \
 && useradd --system --uid 10001 --gid planeai --no-create-home --shell /usr/sbin/nologin planeai \
 # A new named volume copies this directory's ownership, so uid 10001 can write to it.
 && mkdir -p /models && chown 10001:10001 /models

COPY init_model.py /opt/model-init/init_model.py

USER 10001:10001
WORKDIR /models
ENTRYPOINT ["python", "/opt/model-init/init_model.py"]
