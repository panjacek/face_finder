# Face detection model choice

Why this project downloads a `.tflite` file, which one it uses, and what the
alternatives were. Decided 2026-08-25 as part of the mediapipe 1.x migration
(see `docs/plans/` for session plans).

## Why a model file is needed at all

`mediapipe` used to ship both an inference engine **and** the face-detection
neural nets inside its wheels. Code like
`mp.solutions.face_detection.FaceDetection(model_selection=1)` simply pointed
at a file already sitting in `site-packages`
(`mediapipe/modules/face_detection/face_detection_full_range.tflite`) — no
download ever happened.

That ended in two steps (verified against PyPI registry metadata):

| mediapipe release | Solutions API + bundled models | wheels |
| --- | --- | --- |
| ≤ 0.10.21 | yes — 0.10.21 is the **last one ever** | cp39–cp312 only |
| *(0.10.22 – 0.10.29)* | *never published to PyPI* | — |
| ≥ 0.10.30 | removed entirely; Tasks API only | `py3-none` |

mediapipe 1.x is engine-only: `FaceDetector` demands a model path via
`BaseOptions(model_asset_path=...)` and crashes at startup without one. The
one-time fetch below replaces what pip used to carry.

## Chosen model

[BlazeFace (full-range), float16](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite)

- official Google AI Edge Tasks-format model for the Face Detector task
- 1,083,786 bytes (~1.0 MB); md5 `XeN2/MhVJzxccgdm02UjoA==` (verified live
  2026-08-25; pinned in `scripts/fetch_model.py`)
- full-range = same detection range class as today's
  `model_selection=1` → no recall regression on distant/small faces
- output: pixel-coordinate bounding boxes + confidence, plus 6 facial keypoints
  (verified at migration time); annotated output currently draws box + score
  label only

## Alternatives considered

| option | size | far-face recall | verdict |
| --- | --- | --- | --- |
| BlazeFace short-range (official) | ~0.23 MB | drop vs today | rejected: recall regression |
| YuNet via `cv2.FaceDetectorYN` | 232 KB ONNX, MIT | better on small faces (AP_hard ≈ 0.75) | parked: would delete mediapipe entirely (unlocks numpy 2 / py3.14 instantly) but means a bigger rewrite and loses the Tasks GPU-delegate path; revisit if mediapipe keeps hurting |
| YOLOv8n-face et al. from HF | ≥ 6 MB | best | rejected: heavy, needs conversion to Tasks format, overkill for folder scanning |

## Delivery mechanism

The binary never enters git:

- `models/` is gitignored
- `scripts/fetch_model.py` downloads idempotently (skips when present,
  asserts the md5 pin, fails loudly on mismatch)
- invoked by `make setup`, the CI unit job, and a Dockerfile layer
- `core.py` fails fast with "run make setup" if the file is missing — the
  library itself never touches the network

Fresh clones need `make setup` before anything runs — already true today via
`uv sync`, so no extra ceremony.
