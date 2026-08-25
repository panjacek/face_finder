from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from face_finder import core

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "src" / "web_app" / "app.py"


def _write_noise_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


@pytest.fixture
def image_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_noise_image(tmp_path / "one.png", 7)
    _write_noise_image(tmp_path / "two.jpg", 8)
    monkeypatch.setenv("FACE_FINDER_ROOT", str(tmp_path))
    monkeypatch.setenv("FACE_FINDER_OUTPUT", str(tmp_path / "out"))
    return tmp_path


@pytest.fixture
def empty_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FACE_FINDER_ROOT", str(tmp_path))
    monkeypatch.setenv("FACE_FINDER_OUTPUT", str(tmp_path / "out"))
    return tmp_path


@pytest.fixture
def fake_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def _process(folder: str, out_dir: str | Path) -> int:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(out / f"{Path(folder).name}.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
        return 1

    monkeypatch.setattr(core, "process_folder", _process)


def _boot() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_app_boots_without_folder_selection(image_root: Path) -> None:
    _boot()


def test_app_selecting_folder_shows_view_radio(image_root: Path) -> None:
    at = _boot()
    at.selectbox[0].set_value(str(image_root)).run()

    assert not at.exception
    assert len(at.radio) == 1
    assert at.radio[0].options == ["Raw", "Faces"]
    assert at.radio[0].value is None


def test_app_raw_view_renders_images(image_root: Path) -> None:
    at = _boot()
    at.selectbox[0].set_value(str(image_root)).run()
    at.radio[0].set_value("Raw").run()

    assert not at.exception
    assert len(at.image) == 1
    assert len(at.image[0].value) == 2


def test_app_empty_folder_raw_warns(empty_root: Path) -> None:
    at = _boot()
    at.selectbox[0].set_value(str(empty_root)).run()
    at.radio[0].set_value("Raw").run()

    assert not at.exception
    assert len(at.warning) == 1


def test_app_faces_view_creates_output_dir(image_root: Path, fake_process: None) -> None:
    at = _boot()
    at.selectbox[0].set_value(str(image_root)).run()
    at.radio[0].set_value("Faces").run(timeout=120)

    assert not at.exception
    assert (image_root / "out").is_dir()
    assert len(at.image[0].value) == 1


def test_app_faces_isolated_per_folder(image_root: Path, fake_process: None) -> None:
    sub = image_root / "sub"
    sub.mkdir()
    _write_noise_image(sub / "three.png", 9)

    at = _boot()
    at.selectbox[0].set_value(str(sub)).run()
    at.radio[0].set_value("Faces").run(timeout=120)
    assert len(at.image) == 1

    at.selectbox[0].set_value(str(image_root)).run(timeout=120)

    assert not at.exception
    assert at.radio[0].value == "Faces"
    assert len(at.image) == 1
