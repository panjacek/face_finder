# Face Finder

[![CI](https://github.com/panjacek/face_finder/actions/workflows/ci.yml/badge.svg)](https://github.com/panjacek/face_finder/actions/workflows/ci.yml)

Face detection for folders of images. Walks a directory tree, detects faces
with [MediaPipe Face Detection](https://developers.google.com/mediapipe), and
writes annotated copies (bounding boxes + confidence scores) into an output
folder.

The project started as a fun way to evaluate the MediaPipe Tasks API end to
end: packaging, CLI and web frontends, CI, Docker (including GPU passthrough)
around one small face-detection model.

Two ways to use it:

- **CLI** — batch-process a directory from the terminal
- **Web UI** — Streamlit app: browse a folder, toggle face detection, page
  through thumbnails

## Quick start

Requires [uv](https://docs.astral.sh/uv/) only (Python 3.14 is fetched
automatically; cv2 needs system `libgl1`/`libglib2.0-0`, see `Dockerfile`).

You also need photos containing faces — real ones or AI-generated samples,
any folder works. Everything runs on your machine: the only network access is
a one-time ~1 MB checksum-pinned model download during `make setup`. After
that, CLI and web UI are fully offline.

```bash
make setup                       # uv sync + download the face-detection model (see docs/model_choice.md)
make run-cli CALC_ARGS="path/to/images"  # process a folder
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
| -d, --device | inference device: `cpu` or `gpu` [cpu] |

## Web UI

```bash
make run-web        # http://localhost:8501
```

Pick a folder under the app root (`FACE_FINDER_ROOT`, default `/app` inside
docker), then either view raw images or toggle **Show Faces** to run
detection over the selection. Detection results are cached and re-run only
when folder contents change. Set `FACE_FINDER_DEVICE=gpu` to detect on the
GPU delegate.

## GPU

CPU is the default device. `-d gpu` (CLI) or `FACE_FINDER_DEVICE=gpu` (web)
selects the MediaPipe GPU delegate; if delegate creation fails on a host,
processing falls back to CPU with a warning.

```bash
face_finder path/to/images -d gpu   # bare metal (needs libegl1/libgles2)
USE_GPU=1 make run                  # container via nvidia-container-toolkit (NVIDIA GPU)
```
AMD iGPU is bare-metal only: run `face_finder ... -d gpu` directly; no container
passthrough path is provided.

Measured numbers and methodology: [docs/benchmarks.md](docs/benchmarks.md).

## Docker

Same image drives three compose services:

```bash
IMAGES=/path/to/images make run     # one-shot CLI in container
USE_GPU=1 make run                  # NVIDIA runtime variant
make up                             # web UI on :8080
```

## Development

```bash
make help         # annotated target list
make lint         # ruff + mypy + bandit
make format       # auto-format
make test         # unit, smoke and Streamlit app tests (coverage); gpu excluded
make test-gpu     # container smoke test (NVIDIA GPU host only)
```

Layout: `src/face_finder/` package (library + CLI), `src/web_app/` Streamlit
app, `tests/` pytest suite. CI runs lint → unit (+ Streamlit app tests) ∥
docker-build on every push/PR.

## License

Unlicense — see [LICENSE](LICENSE).
