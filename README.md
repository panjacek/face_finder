# Face Finder

[![CI](https://github.com/panjacek/face_finder/actions/workflows/ci.yml/badge.svg)](https://github.com/panjacek/face_finder/actions/workflows/ci.yml)

Face detection for folders of images. Walks a directory tree, detects faces
with [MediaPipe Face Detection](https://developers.google.com/mediapipe), and
writes annotated copies (bounding boxes + keypoints) into an output folder.

Two ways to use it:

- **CLI** — batch-process a directory from the terminal
- **Web UI** — Streamlit app: browse a folder, toggle face detection, page
  through thumbnails

## Quick start

Requires [uv](https://docs.astral.sh/uv/) only (Python 3.12 is fetched
automatically; cv2 needs system `libgl1`/`libglib2.0-0`, see `Dockerfile`).

```bash
make setup                       # uv sync --group dev --group web
make run-cli CALC_ARGS="DB_jpg"  # process a folder
```

Annotated images land in `./output/` (override with `-o`).

## CLI usage

```
face_finder [-h] [-l LOG_LVL] [-o OUTPUT_DIR] source
```

| argument | description |
| -------- | ----------- |
| source (positional) | Directory or single image file to parse |
| -l, --log_lvl | logger level to display [INFO] |
| -o, --output_dir | dir for annotated images [output] |

## Web UI

```bash
make run-web        # http://localhost:8501
```

Pick a folder under the app root (`FACE_FINDER_ROOT`, default `/app` inside
docker), then either view raw images or toggle **Show Faces** to run
detection over the selection.

## Docker

Same image drives three compose services:

```bash
IMAGES=/path/to/images make run     # one-shot CLI in container
USE_GPU=1 make run                  # GPU device passthrough variant
make up                             # web UI on :8080
```

## Development

```bash
make help         # annotated target list
make lint         # ruff + mypy + bandit
make format       # auto-format
make test         # unit + detection smoke tests (coverage)
make test-web     # Streamlit app tests (no browser needed)
```

Layout: `src/face_finder/` package (library + CLI), `src/web_app/` Streamlit
app, `tests/` pytest suite. CI runs lint → unit (+ Streamlit app tests) ∥
docker-build on every push/PR.

<!-- TODO: demo GIF of the web UI -->

## License

Unlicense — see [LICENSE](LICENSE).
