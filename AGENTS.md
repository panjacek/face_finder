# AGENTS.md

Conventions for working in this repo.

## Layout

- `src/face_finder/` — installable package (library + CLI). Entry point:
  `face_finder = face_finder.core:main`.
- `src/web_app/` — Streamlit app, not part of the wheel.
- `tests/` — unit + smoke + web tests, plain pytest.
- Test material (`DB_jpg/`, `fake_set1/`, `gallery*`, `random_multi/`) stays
  on disk and out of git. Never delete or move it.

## Toolchain

- uv only: `make setup` (= `uv sync --group dev --group web`, then
  `scripts/fetch_model.py` downloads the face-detection model into gitignored
  `models/` — see docs/model_choice.md). Python pinned to 3.14 via
  `.python-version`.
- Lint/format: ruff (line length 120), mypy (`disallow_untyped_defs`),
  bandit. `make lint` / `make format`.

## Code rules

- NEVER import inside functions/methods — all imports at module top level.
- Type hints on every function signature.
- No comments unless explaining a non-obvious domain decision.
- cv2/mediapipe need system `libgl1 libglib2.0-0` on the host (see Dockerfile).

## Verification

```bash
make lint && make test        # static + all uv-runnable tests (unit/smoke/web; gpu excluded)
make test-gpu                 # container smoke test with /dev/dri mounted (GPU host only)
```
