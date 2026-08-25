"""Fetch the face-detection model into gitignored models/.

Idempotent: skips when a checksum-matching file is present, re-downloads on
mismatch, fails loudly on corrupted downloads. Rationale: docs/model_choice.md.
"""

import base64
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)
EXPECTED_MD5_B64 = "XeN2/MhVJzxccgdm02UjoA=="  # pinned from Google bucket headers, see docs/model_choice.md

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / MODEL_URL.rsplit("/", 1)[-1]


def _md5_b64(path: Path) -> str:
    return base64.b64encode(hashlib.md5(path.read_bytes(), usedforsecurity=False).digest()).decode()


def fetch_model() -> None:
    if MODEL_PATH.exists():
        if _md5_b64(MODEL_PATH) == EXPECTED_MD5_B64:
            print(f"model already present: {MODEL_PATH}")
            return
        print(f"checksum mismatch on {MODEL_PATH}, re-downloading")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part_path = MODEL_PATH.with_name(MODEL_PATH.name + ".part")
    print(f"downloading {MODEL_URL} -> {MODEL_PATH}")
    urlretrieve(MODEL_URL, part_path)
    actual = _md5_b64(part_path)
    if actual != EXPECTED_MD5_B64:
        part_path.unlink()
        raise SystemExit(f"downloaded model failed checksum: got {actual}, expected {EXPECTED_MD5_B64}")
    part_path.replace(MODEL_PATH)
    print(f"ok ({MODEL_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    try:
        fetch_model()
    except OSError as exc:
        raise SystemExit(f"model download failed: {exc}") from exc
