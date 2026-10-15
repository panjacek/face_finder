# syntax=docker/dockerfile:1
ARG BASE_IMAGE=python:3.14-slim
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libegl-mesa0 \
        libgbm1 \
        libgl1-mesa-dri \
        libgles2 \
        ocl-icd-libopencl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --group web

COPY scripts/fetch_model.py ./scripts/fetch_model.py
RUN python scripts/fetch_model.py

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group web

RUN useradd --uid 1000 --create-home --shell /bin/bash app
USER app

EXPOSE 8501
CMD ["face_finder", "--help"]
