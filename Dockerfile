ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

RUN apt-get update\
    && apt-get install -y --no-install-recommends\
        libgl1 \
        libglib2.0-0 \
    && apt-get clean all

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --group web

COPY src ./src
RUN uv sync --frozen --no-dev --group web

EXPOSE 8501
