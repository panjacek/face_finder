import base64
import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fetch_model as fm  # noqa: E402

MODEL_BYTES = b"tflite-model-bytes"


def _md5_b64(content: bytes) -> str:
    return base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode()


@pytest.fixture
def model_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "models" / "blaze_face_full_range.tflite"
    monkeypatch.setattr(fm, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(fm, "MODEL_PATH", target)
    monkeypatch.setattr(fm, "EXPECTED_MD5_B64", _md5_b64(MODEL_BYTES))
    return target


def _forbid_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fm, "urlretrieve", lambda *args: pytest.fail("download attempted"))


def test_fetch_skips_when_checksum_matches(model_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_paths.parent.mkdir(parents=True)
    model_paths.write_bytes(MODEL_BYTES)
    _forbid_download(monkeypatch)

    fm.fetch_model()

    assert model_paths.read_bytes() == MODEL_BYTES


def test_fetch_redownloads_on_checksum_mismatch(model_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_paths.parent.mkdir(parents=True)
    model_paths.write_bytes(b"corrupted")

    def fake_urlretrieve(url: str, part_path: str) -> None:
        Path(part_path).write_bytes(MODEL_BYTES)

    monkeypatch.setattr(fm, "urlretrieve", fake_urlretrieve)

    fm.fetch_model()

    assert model_paths.read_bytes() == MODEL_BYTES
    assert not model_paths.with_name(model_paths.name + ".part").exists()


def test_fetch_fails_loudly_on_corrupt_download(model_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fm, "urlretrieve", lambda url, part_path: Path(part_path).write_bytes(b"junk"))

    with pytest.raises(SystemExit, match="checksum"):
        fm.fetch_model()

    assert not model_paths.exists()
    assert not model_paths.with_name(model_paths.name + ".part").exists()


def test_real_model_file_matches_pin() -> None:
    if not fm.MODEL_PATH.exists():
        pytest.skip("model not fetched on this host")
    assert fm._md5_b64(fm.MODEL_PATH) == fm.EXPECTED_MD5_B64
