ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

RUN apt-get update\
    && apt-get install -y --no-install-recommends\
        libgl1 \
        libglib2.0-0 \
    && apt-get clean all

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app.py /app/app.py
COPY ./face_finder /app/face_finder
COPY ./setup.py /app/setup.py
COPY ./pages /app/pages
COPY ./helpers /app/helpers

RUN pip install --no-cache-dir -e /app

EXPOSE 8501
